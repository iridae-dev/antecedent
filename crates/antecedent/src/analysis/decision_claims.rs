//! Typed conversions from the 2.2 estimate-crate claim types into decision atoms.
//!
//! The design crate cannot depend on the estimate crate, so the conversions live
//! here, beside [`BoundDecisionContract`]. Each reads a real 2.2 result and
//! produces what the design adapters consume:
//!
//! - a [`ScenarioSetReport`] becomes one structural atom per scenario, keeping
//!   the scenario's status (`identified`, `structurally_unidentified`,
//!   `missing_evidence`, `unsupported_provider`, `support_failure`,
//!   `not_certified`, `unevaluated`), its declared weight and its support. A
//!   structurally unidentified scenario is an unidentified atom; every other
//!   scenario that produced no law is an unevaluated atom. Their mass is
//!   retained and never renormalized over the identified scenarios;
//! - declared weights are used only when the report is weighted
//!   ([`WeightedScenarioReport`]); an unweighted report yields unweighted atoms;
//! - a [`CpdagScenarioReport`] is graph dependent: one atom per completion, no
//!   weights, since completion counts are never probabilities. Completions a
//!   stop never enumerated become one unevaluated atom that carries the count,
//!   so invariance is never claimed over a class that was not fully read;
//! - an identified set (a [`StructuralEnvelope`], a [`WeightedScenarioReport`]
//!   range or an `IdentifiedSetInterval`) becomes one interval per action for
//!   `evaluate_identified_sets`, by interval arithmetic over the action's
//!   inputs.
//!
//! Every conversion records the source identity it read: the scenario ids, a
//! content digest of each scenario's status, weight and law, and the premises
//! and data digests the caller retained. A conversion given an expected
//! [`ClaimSource`] refuses when any of them changed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::result_large_err)]

use std::collections::HashSet;

use antecedent_core::{
    ExternalRefusal, IdentifiedSet, ScientificQuantity, SupportStatus, VariableId,
};
use antecedent_design::decision_adapters::{
    AdaptedClaims, AdaptedDecision, AdapterError, ClaimKind, ClaimProbability, IdentifiedUtility,
    SuppliedClaim, adapt_finite_scenarios, adapt_graph_dependent_claims, utility_interval,
};
pub use antecedent_design::decision_contract::StructuralPolicy as DecisionStructuralPolicy;
use antecedent_design::decision_contract::{AdmissibleDecisionContract, StructuralPolicy};
use antecedent_design::decision_eval::Verdict;
use antecedent_design::decision_robustness::AtomSupport;
use antecedent_design::decision_structural::{AtomEvidence, AtomStatus};
use antecedent_estimate::IdentifiedSetInterval;
use antecedent_estimate::cpdag_scenarios::CpdagScenarioReport;
use antecedent_estimate::transport_scenarios::{
    SCENARIO_STATUSES, ScenarioResult, ScenarioSetReport, StatusMass, StructuralEnvelope,
    WeightedScenarioReport,
};
use antecedent_expr::ExactDistribution;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

use super::decision_contract_facade::{BoundDecisionContract, DecisionFacadeRefusal};

/// Atom identity of the completions a stop never enumerated.
pub const NOT_ENUMERATED_ATOM: &str = "cpdag.not_enumerated";

/// Why a conversion refused.
#[derive(Clone, Debug, PartialEq)]
pub enum ClaimError {
    /// The 2.2 result cannot be read as the requested claim.
    UnsupportedClaimShape(&'static str),
    /// An action input has no bound outcome or range to read a utility from.
    MissingScenarioUtility {
        /// Action whose utility cannot be formed.
        action: String,
        /// Input position that has no bound value.
        input: usize,
    },
    /// `BayesOverStructures` needs genuine declared weights.
    ProbabilitiesNotDeclared {
        /// The structures are completions, whose counts are never probabilities.
        completion_counts: bool,
    },
    /// The source differs from the identity the caller retained.
    IdentityMismatch(String),
    /// A weighted report has no range (an unbounded outcome domain with
    /// unaccounted mass, or no identified scenario).
    RangeUnavailable,
    /// Structures that no identified set covers: unidentified or unevaluated
    /// scenarios, or a truncated completion enumeration.
    UnresolvedScenarios(usize),
    /// A design adapter refused.
    Adapter(AdapterError),
}

impl ClaimError {
    /// Structured refusal with a registered code and a namespaced detail.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = antecedent_core::reason_code!("invalid_argument");
        let unsatisfied = antecedent_core::reason_code!("decision_contract_unsatisfied");
        let build = |code, detail: &str, offending: Option<String>, remedy| ExternalRefusal {
            code,
            stage: "adapt",
            detail: detail.to_owned(),
            offending,
            expected: None,
            supplied: None,
            capability: None,
            remedy,
        };
        match self {
            Self::UnsupportedClaimShape(why) => ExternalRefusal {
                expected: Some((*why).to_owned()),
                ..build(invalid, "decision_claims.unsupported_claim_shape", None, None)
            },
            Self::MissingScenarioUtility { action, input } => build(
                unsatisfied,
                "decision_claims.missing_scenario_utility",
                Some(format!("{action}.inputs[{input}]")),
                Some("bind every action input to an outcome the scenarios evaluate"),
            ),
            Self::ProbabilitiesNotDeclared { completion_counts } => ExternalRefusal {
                expected: Some("declared scenario weights".to_owned()),
                supplied: Some(
                    (if *completion_counts {
                        "completion counts (never probabilities)"
                    } else {
                        "none"
                    })
                    .to_owned(),
                ),
                ..build(
                    unsatisfied,
                    "decision_claims.probabilities_not_declared",
                    None,
                    Some(
                        "declare weights for the scenario set, or use a worst-case or invariance policy",
                    ),
                )
            },
            Self::IdentityMismatch(part) => build(
                invalid,
                "decision_claims.identity_mismatch",
                Some(part.clone()),
                Some("convert the same scenario report the identity was recorded from"),
            ),
            Self::RangeUnavailable => build(
                unsatisfied,
                "decision_claims.range_unavailable",
                None,
                Some("declare a finite outcome domain or weights that account for every scenario"),
            ),
            Self::UnresolvedScenarios(count) => build(
                unsatisfied,
                "decision_claims.unresolved_scenarios",
                Some(count.to_string()),
                Some(
                    "an identified set covers identified scenarios only; resolve the rest or use atoms",
                ),
            ),
            Self::Adapter(error) => error.to_refusal(),
        }
    }
}

/// An outcome the scenarios evaluate, bound to the action input it answers.
#[derive(Clone, Debug, PartialEq)]
pub struct OutcomeBinding {
    /// Outcome coordinate of the scenario law.
    pub outcome: VariableId,
    /// Scientific coordinate an action input reads it as.
    pub quantity: ScientificQuantity,
}

/// What the conversion needs beyond the 2.2 report: the quantity each outcome
/// answers, the lineage of the laws, and the support of each scenario.
#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioClaimBinding {
    /// Outcomes in artifact column order.
    pub outcomes: Vec<OutcomeBinding>,
    /// Meaning of the scenario laws.
    pub semantic: DistributionMeaningWire,
    /// Calibration of the laws: `Exact` for supplied exact laws, `Unmeasured`
    /// (or `PointOnly`) for an empirical plug-in.
    pub calibration: DistributionCalibration,
    /// Stable source or study identity.
    pub source_id: String,
    /// Provider identity (the scenario set's provider).
    pub provider_id: String,
    /// Identification identity of the causal contract the laws answer.
    pub causal_contract_id: String,
    /// Premises digest of the scenario set, as the caller retained it.
    pub premises_digest: String,
    /// Data digest of the scenario set, as the caller retained it.
    pub data_digest: String,
    /// Support of an identified scenario's evidence, by scenario id.
    pub support: Vec<(String, AtomSupport)>,
    /// Support of an identified scenario without its own entry.
    pub default_support: AtomSupport,
}

impl ScenarioClaimBinding {
    /// Bind `outcomes` with exact interventional laws and unassessed support.
    #[must_use]
    pub fn new(
        outcomes: Vec<OutcomeBinding>,
        source_id: &str,
        provider_id: &str,
        causal_contract_id: &str,
        premises_digest: &str,
        data_digest: &str,
    ) -> Self {
        Self {
            outcomes,
            semantic: DistributionMeaningWire::InterventionalPredictive,
            calibration: DistributionCalibration::Exact,
            source_id: source_id.to_owned(),
            provider_id: provider_id.to_owned(),
            causal_contract_id: causal_contract_id.to_owned(),
            premises_digest: premises_digest.to_owned(),
            data_digest: data_digest.to_owned(),
            support: Vec::new(),
            default_support: AtomSupport::unassessed(),
        }
    }

    /// Set the support of identified scenarios without their own entry.
    #[must_use]
    pub fn with_default_support(mut self, support: AtomSupport) -> Self {
        self.default_support = support;
        self
    }

    /// Set the support of one scenario's evidence.
    #[must_use]
    pub fn with_support(mut self, scenario: &str, support: AtomSupport) -> Self {
        self.support.push((scenario.to_owned(), support));
        self
    }

    fn support_of(&self, scenario: &str) -> AtomSupport {
        self.support
            .iter()
            .find(|(id, _)| id == scenario)
            .map_or_else(|| self.default_support.clone(), |(_, support)| support.clone())
    }

    fn validate(&self) -> Result<(), ClaimError> {
        let blank = [
            &self.source_id,
            &self.provider_id,
            &self.causal_contract_id,
            &self.premises_digest,
            &self.data_digest,
        ]
        .into_iter()
        .any(|s| s.trim().is_empty());
        let mut seen = HashSet::new();
        let repeated = self.outcomes.iter().any(|b| !seen.insert(b.outcome));
        if blank || self.outcomes.is_empty() || repeated {
            return Err(ClaimError::UnsupportedClaimShape(
                "a binding needs non-blank identities and distinct bound outcomes",
            ));
        }
        Ok(())
    }
}

/// Content digest of one scenario as the conversion read it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScenarioDigest {
    /// Scenario (or completion) identity.
    pub id: String,
    /// Status the scenario reported.
    pub status: String,
    /// Digest of its id, status, declared weight and law.
    pub digest: String,
}

/// The identity of what a conversion read.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClaimSource {
    /// `transport_scenarios` or `cpdag_completions`.
    pub kind: &'static str,
    /// Identity of the supplied CPDAG, for completions.
    pub cpdag_identity: Option<String>,
    /// Premises digest the caller retained.
    pub premises_digest: String,
    /// Data digest the caller retained.
    pub data_digest: String,
    /// Every scenario in report order.
    pub scenarios: Vec<ScenarioDigest>,
    /// Digest of everything above.
    pub identity: String,
}

const HEX: &[u8; 16] = b"0123456789abcdef";

fn hex(bytes: &[u8; 32]) -> String {
    bytes
        .iter()
        .flat_map(|b| [HEX[usize::from(b >> 4)] as char, HEX[usize::from(b & 0x0f)] as char])
        .collect()
}

fn put_len(bytes: &mut Vec<u8>, len: usize) {
    bytes.extend(u64::try_from(len).unwrap_or(u64::MAX).to_le_bytes());
}

fn put_str(bytes: &mut Vec<u8>, text: &str) {
    put_len(bytes, text.len());
    bytes.extend(text.as_bytes());
}

fn put_f64(bytes: &mut Vec<u8>, value: f64) {
    bytes.extend(value.to_bits().to_le_bytes());
}

fn scenario_digest(result: &ScenarioResult) -> String {
    let mut bytes = Vec::new();
    put_str(&mut bytes, &result.name);
    put_str(&mut bytes, result.status);
    match result.weight {
        Some(weight) => {
            bytes.push(1);
            put_f64(&mut bytes, weight);
        }
        None => bytes.push(0),
    }
    match &result.distribution {
        Some(law) => {
            bytes.push(1);
            put_len(&mut bytes, law.outcomes.len());
            for outcome in law.outcomes.iter() {
                bytes.extend(outcome.raw().to_le_bytes());
            }
            put_len(&mut bytes, law.atoms.len());
            for (atom, probability) in law.atoms.iter().zip(law.probabilities.iter()) {
                for value in atom.iter() {
                    match value.as_f64() {
                        Some(x) => {
                            bytes.push(1);
                            put_f64(&mut bytes, x);
                        }
                        None => bytes.push(0),
                    }
                }
                put_f64(&mut bytes, *probability);
            }
        }
        None => bytes.push(0),
    }
    hex(&antecedent_io::payload_digest("decision_claims_scenario", &bytes))
}

impl ClaimSource {
    fn new(
        kind: &'static str,
        cpdag_identity: Option<String>,
        binding: &ScenarioClaimBinding,
        scenarios: Vec<ScenarioDigest>,
    ) -> Self {
        let mut bytes = Vec::new();
        put_str(&mut bytes, kind);
        put_str(&mut bytes, cpdag_identity.as_deref().unwrap_or(""));
        put_str(&mut bytes, &binding.premises_digest);
        put_str(&mut bytes, &binding.data_digest);
        put_len(&mut bytes, scenarios.len());
        for scenario in &scenarios {
            put_str(&mut bytes, &scenario.id);
            put_str(&mut bytes, &scenario.status);
            put_str(&mut bytes, &scenario.digest);
        }
        let identity = hex(&antecedent_io::payload_digest("decision_claims_source", &bytes));
        Self {
            kind,
            cpdag_identity,
            premises_digest: binding.premises_digest.clone(),
            data_digest: binding.data_digest.clone(),
            scenarios,
            identity,
        }
    }

    /// Check this source against the identity the caller retained.
    ///
    /// # Errors
    /// The first part that differs: the kind, the CPDAG, the premises digest,
    /// the data digest, the scenario list, or one scenario's digest.
    pub fn verify(&self, expected: &Self) -> Result<(), ClaimError> {
        let mismatch = |part: &str| Err(ClaimError::IdentityMismatch(part.to_owned()));
        if self.kind != expected.kind {
            return mismatch("kind");
        }
        if self.cpdag_identity != expected.cpdag_identity {
            return mismatch("cpdag_identity");
        }
        if self.premises_digest != expected.premises_digest {
            return mismatch("premises_digest");
        }
        if self.data_digest != expected.data_digest {
            return mismatch("data_digest");
        }
        let ids = |s: &Self| s.scenarios.iter().map(|d| d.id.clone()).collect::<Vec<_>>();
        if ids(self) != ids(expected) {
            return mismatch("scenario_set");
        }
        for (read, wanted) in self.scenarios.iter().zip(&expected.scenarios) {
            if read.digest != wanted.digest || read.status != wanted.status {
                return mismatch(&read.id);
            }
        }
        if self.identity != expected.identity {
            return mismatch("identity");
        }
        Ok(())
    }
}

/// One scenario as an atom: what became of it and the support it carries.
#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioAtomRecord {
    /// Scenario (or completion) identity; also the atom identity.
    pub id: String,
    /// One of `SCENARIO_STATUSES`, or `unevaluated` for the not-enumerated atom.
    pub status: &'static str,
    /// Declared weight, when the report is weighted.
    pub weight: Option<f64>,
    /// Overall empirical support of the atom's evidence.
    pub support: SupportStatus,
    /// Content digest of the scenario as read.
    pub digest: String,
    /// Why the scenario produced no law, when it did not.
    pub detail: Option<String>,
}

/// Converted scenario claims with their source identity and retained masses.
#[derive(Clone, Debug)]
pub struct ScenarioClaims {
    /// What was read.
    pub source: ClaimSource,
    /// Atoms for the contract's structural policy.
    pub adapted: AdaptedClaims,
    /// Per-atom status, weight, support and digest, in atom order.
    pub atoms: Vec<ScenarioAtomRecord>,
    /// Whether the report declared weights.
    pub declared_weights: bool,
    /// Count and declared mass per status, exactly as the report gives them.
    pub masses: Vec<StatusMass>,
    /// Declared mass assigned to no scenario; `None` for an unweighted report.
    pub residual_mass: Option<f64>,
}

/// An evaluated decision with the claims it read.
#[derive(Clone, Debug)]
pub struct ScenarioDecision {
    /// The converted claims.
    pub claims: ScenarioClaims,
    /// The decision over them.
    pub decision: AdaptedDecision,
    /// Each evaluated atom's leading actions (empty when no action is admissible).
    pub leaders: Vec<(String, Vec<String>)>,
}

/// Each evaluated atom (scenario or graph) with the actions that lead in it.
#[must_use]
pub fn atom_leaders(decision: &AdaptedDecision) -> Vec<(String, Vec<String>)> {
    decision
        .structural
        .atoms
        .iter()
        .filter_map(|atom| match &atom.status {
            AtomStatus::Evaluated(result) => Some((
                atom.id.clone(),
                match &result.verdict {
                    Verdict::UniquelyOptimal(id) => vec![id.clone()],
                    Verdict::Indistinguishable(ids) => ids.clone(),
                    Verdict::NoAdmissibleAction => Vec::new(),
                },
            )),
            _ => None,
        })
        .collect()
}

const SHAPE_LAW: &str =
    "an identified scenario law must be finite and numeric over the bound outcomes";

fn law_artifact(
    law: &ExactDistribution,
    binding: &ScenarioClaimBinding,
) -> Result<Box<DistributionArtifact>, ClaimError> {
    let shape = ClaimError::UnsupportedClaimShape(SHAPE_LAW);
    let axes = binding
        .outcomes
        .iter()
        .map(|b| law.outcomes.iter().position(|o| *o == b.outcome))
        .collect::<Option<Vec<usize>>>()
        .ok_or_else(|| shape.clone())?;
    if law.atoms.is_empty() || law.atoms.len() != law.probabilities.len() {
        return Err(shape);
    }
    let mut draws = Vec::with_capacity(law.atoms.len() * axes.len());
    for atom in law.atoms.iter() {
        for axis in &axes {
            let value = atom
                .get(*axis)
                .and_then(antecedent_core::Value::as_f64)
                .ok_or_else(|| shape.clone())?;
            draws.push(value);
        }
    }
    let quantities: Vec<ScientificQuantity> =
        binding.outcomes.iter().map(|b| b.quantity.clone()).collect();
    let identity = DistributionIdentity::new(
        binding.semantic,
        &quantities,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: binding.source_id.clone(),
            provider_id: binding.provider_id.clone(),
            rng_id: "deterministic_exact".to_owned(),
            snapshot_id: binding.data_digest.clone(),
            causal_contract_id: binding.causal_contract_id.clone(),
        },
    )
    .map_err(|_| shape.clone())?;
    let artifact = DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".to_owned(), "quantity".to_owned()],
            shape: [law.atoms.len(), axes.len()],
            weights: Some(law.probabilities.to_vec()),
            supported: None,
            calibration: binding.calibration,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .map_err(|_| shape)?;
    Ok(Box::new(artifact))
}

fn evidence_of(
    result: &ScenarioResult,
    binding: &ScenarioClaimBinding,
) -> Result<AtomEvidence, ClaimError> {
    Ok(match result.status {
        "identified" => {
            let law = result.distribution.as_ref().ok_or(ClaimError::UnsupportedClaimShape(
                "an identified scenario carries its law",
            ))?;
            AtomEvidence::Evaluated(law_artifact(law, binding)?)
        }
        "structurally_unidentified" => AtomEvidence::Unidentified,
        _ => AtomEvidence::Unevaluated(match &result.detail {
            Some(detail) => format!("{}: {detail}", result.status),
            None => result.status.to_owned(),
        }),
    })
}

fn support_of(result: &ScenarioResult, binding: &ScenarioClaimBinding) -> AtomSupport {
    match result.status {
        "identified" => binding.support_of(&result.name),
        "support_failure" => AtomSupport::uniform(SupportStatus::OutsideEmpiricalSupport),
        _ => AtomSupport::uniform(SupportStatus::MissingEvidence),
    }
}

struct Input<'a> {
    kind: ClaimKind,
    cpdag_identity: Option<String>,
    results: &'a [ScenarioResult],
    weighted: Option<&'a WeightedScenarioReport>,
    masses: Vec<StatusMass>,
    residual_mass: Option<f64>,
    not_enumerated: usize,
}

/// Declared weights: all scenarios or none, and only with a weighted report.
fn check_weights(input: &Input<'_>) -> Result<bool, ClaimError> {
    let Some(weighted) = input.weighted else {
        if input.results.iter().any(|r| r.weight.is_some()) {
            return Err(ClaimError::UnsupportedClaimShape(
                "scenario weights are declared only through a weighted report",
            ));
        }
        return Ok(false);
    };
    if input.results.iter().any(|r| r.weight.is_none()) {
        return Err(ClaimError::UnsupportedClaimShape(
            "a weighted report needs a declared weight for every scenario",
        ));
    }
    let identified: f64 =
        input.results.iter().filter(|r| r.status == "identified").filter_map(|r| r.weight).sum();
    if (identified - weighted.identified_mass).abs() > 1e-9 {
        return Err(ClaimError::IdentityMismatch("weighted_report".to_owned()));
    }
    Ok(true)
}

fn digests_of(input: &Input<'_>) -> Result<Vec<ScenarioDigest>, ClaimError> {
    let mut digests = Vec::with_capacity(input.results.len() + 1);
    for result in input.results {
        if !SCENARIO_STATUSES.contains(&result.status) {
            return Err(ClaimError::UnsupportedClaimShape("an unknown scenario status"));
        }
        digests.push(ScenarioDigest {
            id: result.name.to_string(),
            status: result.status.to_owned(),
            digest: scenario_digest(result),
        });
    }
    if input.not_enumerated > 0 {
        let mut bytes = Vec::new();
        put_len(&mut bytes, input.not_enumerated);
        digests.push(ScenarioDigest {
            id: NOT_ENUMERATED_ATOM.to_owned(),
            status: "unevaluated".to_owned(),
            digest: hex(&antecedent_io::payload_digest("decision_claims_not_enumerated", &bytes)),
        });
    }
    Ok(digests)
}

fn build(
    input: &Input<'_>,
    binding: &ScenarioClaimBinding,
    policy: StructuralPolicy,
    expected: Option<&ClaimSource>,
) -> Result<ScenarioClaims, ClaimError> {
    binding.validate()?;
    let declared = check_weights(input)?;
    let completions = input.kind == ClaimKind::GraphDependent;
    if policy == StructuralPolicy::BayesOverStructures && !declared {
        return Err(ClaimError::ProbabilitiesNotDeclared { completion_counts: completions });
    }
    let digests = digests_of(input)?;
    let kind = if completions { "cpdag_completions" } else { "transport_scenarios" };
    let source = ClaimSource::new(kind, input.cpdag_identity.clone(), binding, digests);
    if let Some(expected) = expected {
        source.verify(expected)?;
    }
    let mut records = Vec::with_capacity(source.scenarios.len());
    let mut claims = Vec::with_capacity(source.scenarios.len());
    for (result, scenario) in input.results.iter().zip(&source.scenarios) {
        let support = support_of(result, binding);
        records.push(ScenarioAtomRecord {
            id: scenario.id.clone(),
            status: result.status,
            weight: result.weight,
            support: support.overall,
            digest: scenario.digest.clone(),
            detail: result.detail.clone(),
        });
        claims.push(SuppliedClaim {
            id: scenario.id.clone(),
            probability: result
                .weight
                .map_or(ClaimProbability::Unspecified, ClaimProbability::Genuine),
            evidence: evidence_of(result, binding)?,
            support,
        });
    }
    if input.not_enumerated > 0 {
        let count = u64::try_from(input.not_enumerated).unwrap_or(u64::MAX);
        let support = AtomSupport::uniform(SupportStatus::MissingEvidence);
        records.push(ScenarioAtomRecord {
            id: NOT_ENUMERATED_ATOM.to_owned(),
            status: "unevaluated",
            weight: None,
            support: support.overall,
            digest: source.scenarios.last().map(|d| d.digest.clone()).unwrap_or_default(),
            detail: Some(format!("up to {count} completions were never enumerated")),
        });
        claims.push(SuppliedClaim {
            id: NOT_ENUMERATED_ATOM.to_owned(),
            probability: ClaimProbability::CompletionCount(count),
            evidence: AtomEvidence::Unevaluated(format!(
                "not_enumerated: up to {count} completions were never enumerated"
            )),
            support,
        });
    }
    let adapted = if completions {
        adapt_graph_dependent_claims(claims, policy)
    } else {
        adapt_finite_scenarios(claims, policy)
    }
    .map_err(ClaimError::Adapter)?;
    Ok(ScenarioClaims {
        source,
        adapted,
        atoms: records,
        declared_weights: declared,
        masses: input.masses.clone(),
        residual_mass: input.residual_mass,
    })
}

/// Convert a scenario-set report into one atom per scenario.
///
/// Weights are the report's declared weights and nothing else; an unweighted
/// report gives unweighted atoms. `expected` is the source identity retained
/// when the claims were first read.
///
/// # Errors
/// An unreadable report or binding, `BayesOverStructures` without declared
/// weights (`decision_claims.probabilities_not_declared`), a source that
/// differs from `expected` (`decision_claims.identity_mismatch`), and any
/// refusal of the design adapter.
pub fn scenario_claims(
    report: &ScenarioSetReport,
    binding: &ScenarioClaimBinding,
    policy: StructuralPolicy,
    expected: Option<&ClaimSource>,
) -> Result<ScenarioClaims, ClaimError> {
    build(
        &Input {
            kind: ClaimKind::FiniteScenarios,
            cpdag_identity: None,
            results: &report.scenarios,
            weighted: report.weighted.as_ref(),
            masses: report.masses.clone(),
            residual_mass: report.residual_mass,
            not_enumerated: 0,
        },
        binding,
        policy,
        expected,
    )
}

/// Convert a CPDAG completion report into graph-dependent atoms: one per
/// completion, never weighted. Completion counts are retained and never become
/// probabilities, so `BayesOverStructures` refuses.
///
/// # Errors
/// As [`scenario_claims`], plus a completion receipt that disagrees with the
/// scenario results, and a report in which nothing was enumerated.
pub fn cpdag_claims(
    report: &CpdagScenarioReport,
    binding: &ScenarioClaimBinding,
    policy: StructuralPolicy,
    expected: Option<&ClaimSource>,
) -> Result<ScenarioClaims, ClaimError> {
    let inner = report.report.as_ref();
    let results = inner.map_or(&[][..], |r| r.scenarios.as_slice());
    let aligned = results.len() == report.completions.len()
        && results
            .iter()
            .zip(&report.completions)
            .all(|(r, c)| r.name == c.id && r.status == c.status);
    if !aligned {
        return Err(ClaimError::IdentityMismatch("completions".to_owned()));
    }
    if results.is_empty() && report.not_enumerated == 0 {
        return Err(ClaimError::UnsupportedClaimShape("no completion was enumerated"));
    }
    build(
        &Input {
            kind: ClaimKind::GraphDependent,
            cpdag_identity: Some(report.cpdag_identity.to_string()),
            results,
            weighted: None,
            masses: inner.map_or_else(Vec::new, |r| r.masses.clone()),
            residual_mass: None,
            not_enumerated: report.not_enumerated,
        },
        binding,
        policy,
        expected,
    )
}

fn finish(
    bound: &BoundDecisionContract,
    claims: ScenarioClaims,
) -> Result<ScenarioDecision, DecisionFacadeRefusal> {
    let decision = bound.evaluate_claims(&claims.adapted)?;
    let leaders = atom_leaders(&decision);
    Ok(ScenarioDecision { claims, decision, leaders })
}

fn check_causal(
    bound: &BoundDecisionContract,
    binding: &ScenarioClaimBinding,
) -> Result<(), DecisionFacadeRefusal> {
    if binding.causal_contract_id == bound.causal_identification() {
        return Ok(());
    }
    Err(ClaimError::IdentityMismatch("causal_contract_id".to_owned()).to_refusal())
}

/// Read a scenario report under a bound decision contract, end to end: convert
/// to atoms, then evaluate them under the contract's structural policy.
///
/// # Errors
/// A binding whose causal identity is not the bound contract's, any refusal of
/// [`scenario_claims`], and any refusal of the structural evaluation or the
/// robustness assessment (including a `policy` different from the contract's).
pub fn evaluate_scenario_decision(
    bound: &BoundDecisionContract,
    report: &ScenarioSetReport,
    binding: &ScenarioClaimBinding,
    policy: StructuralPolicy,
    expected: Option<&ClaimSource>,
) -> Result<ScenarioDecision, DecisionFacadeRefusal> {
    check_causal(bound, binding)?;
    let claims = scenario_claims(report, binding, policy, expected).map_err(|e| e.to_refusal())?;
    finish(bound, claims)
}

/// As [`evaluate_scenario_decision`], over the completions of a CPDAG.
///
/// # Errors
/// As [`evaluate_scenario_decision`], with [`cpdag_claims`].
pub fn evaluate_cpdag_decision(
    bound: &BoundDecisionContract,
    report: &CpdagScenarioReport,
    binding: &ScenarioClaimBinding,
    policy: StructuralPolicy,
    expected: Option<&ClaimSource>,
) -> Result<ScenarioDecision, DecisionFacadeRefusal> {
    check_causal(bound, binding)?;
    let claims = cpdag_claims(report, binding, policy, expected).map_err(|e| e.to_refusal())?;
    finish(bound, claims)
}

/// One interval per action from a lookup of each input's range. Hard
/// constraints cannot be read from an interval, so a contract with any refuses.
fn utilities_with(
    contract: &AdmissibleDecisionContract,
    support: &AtomSupport,
    lookup: impl Fn(&ScientificQuantity) -> Option<(f64, f64)>,
) -> Result<Vec<IdentifiedUtility>, ClaimError> {
    if !contract.contract.constraints.is_empty() {
        return Err(ClaimError::UnsupportedClaimShape(
            "hard constraints need per-scenario evaluation, not an interval",
        ));
    }
    contract
        .contract
        .actions
        .iter()
        .map(|action| {
            let inputs = action
                .inputs
                .iter()
                .enumerate()
                .map(|(k, quantity)| {
                    let missing = || ClaimError::MissingScenarioUtility {
                        action: action.id.clone(),
                        input: k,
                    };
                    let (lower, upper) = lookup(quantity).ok_or_else(missing)?;
                    IdentifiedSet::try_new(lower, upper).map_err(|_| {
                        ClaimError::UnsupportedClaimShape(
                            "interval endpoints must be finite with lower at most upper",
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let utility = utility_interval(&contract.contract, &action.id, &inputs)
                .map_err(ClaimError::Adapter)?;
            Ok(IdentifiedUtility {
                action_id: action.id.clone(),
                utility,
                hard_exclusions: Vec::new(),
                support: support.clone(),
            })
        })
        .collect()
}

fn from_ranges(
    contract: &AdmissibleDecisionContract,
    binding: &ScenarioClaimBinding,
    ranges: &[(VariableId, f64, f64)],
    support: &AtomSupport,
) -> Result<Vec<IdentifiedUtility>, ClaimError> {
    utilities_with(contract, support, |quantity| {
        let bound = binding
            .outcomes
            .iter()
            .find(|b| b.quantity.require_same_coordinate(quantity).is_ok())?;
        ranges.iter().find(|(o, _, _)| *o == bound.outcome).map(|(_, lo, hi)| (*lo, *hi))
    })
}

/// Interval of each action's utility from a structural envelope. The envelope
/// ranges over the identified scenarios only; it says nothing about the rest.
///
/// # Errors
/// An action input with no bound outcome (`missing_scenario_utility`), a
/// contract with hard constraints, and any refusal of `utility_interval`.
pub fn envelope_identified_utilities(
    contract: &AdmissibleDecisionContract,
    envelope: &StructuralEnvelope,
    binding: &ScenarioClaimBinding,
    support: &AtomSupport,
) -> Result<Vec<IdentifiedUtility>, ClaimError> {
    let ranges: Vec<(VariableId, f64, f64)> =
        envelope.means.iter().map(|m| (m.outcome, m.lower, m.upper)).collect();
    from_ranges(contract, binding, &ranges, support)
}

/// Interval of each action's utility from a weighted report: the range the
/// mixture mean takes when unaccounted mass sits anywhere in the declared
/// outcome domain. Never renormalized over the identified scenarios.
///
/// # Errors
/// A report without a range (`range_unavailable`), plus the refusals of
/// [`envelope_identified_utilities`].
pub fn weighted_identified_utilities(
    contract: &AdmissibleDecisionContract,
    weighted: &WeightedScenarioReport,
    binding: &ScenarioClaimBinding,
    support: &AtomSupport,
) -> Result<Vec<IdentifiedUtility>, ClaimError> {
    let ranges = weighted.ranges.as_ref().ok_or(ClaimError::RangeUnavailable)?;
    from_ranges(contract, binding, ranges, support)
}

/// Interval of each action's utility from a scenario report. A weighted report
/// reads its weighted range; an unweighted one reads its envelope and refuses
/// while any scenario is unresolved, since an envelope says nothing about them.
///
/// # Errors
/// `unresolved_scenarios` for an unweighted report with a scenario that is not
/// identified, `range_unavailable` for a report with no identified scenario,
/// plus the refusals of [`envelope_identified_utilities`].
pub fn report_identified_utilities(
    contract: &AdmissibleDecisionContract,
    report: &ScenarioSetReport,
    binding: &ScenarioClaimBinding,
    support: &AtomSupport,
) -> Result<Vec<IdentifiedUtility>, ClaimError> {
    if let Some(weighted) = &report.weighted {
        return weighted_identified_utilities(contract, weighted, binding, support);
    }
    let unresolved = report.scenarios.iter().filter(|s| s.status != "identified").count();
    if unresolved > 0 {
        return Err(ClaimError::UnresolvedScenarios(unresolved));
    }
    let envelope = report.envelope.as_ref().ok_or(ClaimError::RangeUnavailable)?;
    envelope_identified_utilities(contract, envelope, binding, support)
}

/// Which endpoints of an `IdentifiedSetInterval` are read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntervalReading {
    /// The estimated identified set `[bound_lower, bound_upper]`.
    EstimatedBounds,
    /// The interval covering the true effect at its nominal level.
    CoverageInterval,
}

/// Interval of each action's utility from one 2.2 identified-set interval per
/// input quantity.
///
/// # Errors
/// An interval over a truncated completion enumeration
/// (`unresolved_scenarios`), an input with no interval (`missing_scenario_utility`),
/// a contract with hard constraints, and any refusal of `utility_interval`.
pub fn interval_identified_utilities(
    contract: &AdmissibleDecisionContract,
    intervals: &[(ScientificQuantity, IdentifiedSetInterval)],
    reading: IntervalReading,
    support: &AtomSupport,
) -> Result<Vec<IdentifiedUtility>, ClaimError> {
    let truncated = intervals.iter().filter(|(_, i)| i.truncated).count();
    if truncated > 0 {
        return Err(ClaimError::UnresolvedScenarios(truncated));
    }
    utilities_with(contract, support, |quantity| {
        let (_, interval) =
            intervals.iter().find(|(q, _)| q.require_same_coordinate(quantity).is_ok())?;
        Some(match reading {
            IntervalReading::EstimatedBounds => (interval.bound_lower, interval.bound_upper),
            IntervalReading::CoverageInterval => (interval.lower, interval.upper),
        })
    })
}

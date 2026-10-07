//! Identification repair: which proposed studies make a failed contract identify.
//!
//! [`repair`] is a generic facade. A [`RepairFamily`] owns one failed causal
//! contract and the theorem-specific identifier/validator that decides it; the
//! facade applies hypothetical evidence from [`DurableStudyCandidate`]s (one
//! study, or a bounded subset of studies) and asks the family to re-run its own
//! checker on the hypothetical evidence. A variable-name match never repairs
//! anything: the family's theorem decides, population, regime and joint law
//! included.
//!
//! Two families ship here, reusing existing checkers unchanged:
//!
//! * [`TransportRepairFamily`] re-runs the catalog-aware classical transport
//!   identifier ([`antecedent_identify::identify_catalog_transport`], the same
//!   call [`crate::plan_transport_evidence`] makes) on a previewed catalog and
//!   re-verifies the derivation with
//!   [`antecedent_identify::verify_classical_transport`];
//! * [`BackdoorRepairFamily`] re-runs the back-door identifier on the one joint
//!   observational law a study would deliver (an adjustment set must be read
//!   from a single joint law with the treatment and outcome, so two separate
//!   studies never combine) and independently re-verifies the returned set by
//!   d-separation.
//!
//! The subset search charges a [`SearchBudget`] during work (one operation per
//! subset evaluated, the subset size as depth, a live-state estimate as memory)
//! and reports what was explored and what was left unevaluated. Exhaustion is
//! a resource outcome, never impossibility: unevaluated subsets stay
//! `unevaluated`, and a subset larger than the declared depth is never examined
//! and says so.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, sync::Arc};

use antecedent_core::{
    AverageEffectQuery, CausalQuery, EvidenceCatalog, EvidenceCatalogDelta, EvidenceObligation,
    EvidenceObligationKind, EvidenceRegime, ExecutionContext, IdentificationStatus,
    ObligationProvenance, ObligationRecord, ObligationRegime, ObligationScope, SearchBudget,
    SearchLimits, SearchReceipt, SearchStop, VariableId, reason_code,
};
use antecedent_graph::{
    BitSet, DSeparationWorkspace, Dag, DenseNodeId, GraphWorkspace, NodeRef, SelectionDiagram,
};
use antecedent_identify::{
    BackdoorIdentifier, BoundTransportFunctional, CatalogTransportResult, ClassicalTransportQuery,
    IdentificationError, IdentificationWorkspace, SidLimits, identify_catalog_transport,
    verify_classical_transport,
};

use crate::obligation_adapters::{assumption_obligations, transport_result_obligations};
use crate::plan_common::valid_intervention_values;
use crate::study_candidate::DurableStudyCandidate;

/// Largest candidate list one repair request may declare.
pub const REPAIR_MAX_CANDIDATES: usize = 16;
/// Largest subset size (search depth) a repair request may declare.
pub const REPAIR_MAX_DEPTH: usize = 4;
/// Largest operation count a repair request may declare.
pub const REPAIR_MAX_OPERATIONS: usize = 100_000;
/// Largest memory cap, in bytes, a repair request may declare.
pub const REPAIR_MAX_MEMORY_BYTES: u64 = 256 * 1024 * 1024;
/// Most explored or unevaluated subsets a receipt lists; the total is counted.
pub const REPAIR_RECEIPT_REGIONS: usize = 1024;

/// How a candidate subset fared against the family's theorem checker.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum RepairClassification {
    /// The family's checker identified the hypothetical evidence and the
    /// derivation was re-verified.
    VerifiedSufficient,
    /// The checker ran and the hypothetical evidence leaves the contract unmet.
    Insufficient,
    /// The checker could not certify the hypothetical evidence (bounded search,
    /// unsupported shape, an unresolved assumption); no impossibility is implied.
    NotCertified,
    /// The candidate or its hypothetical evidence is malformed.
    Invalid,
    /// A budget stop or cancellation left the subset unexamined.
    Unevaluated,
}

impl RepairClassification {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VerifiedSufficient => "verified_sufficient",
            Self::Insufficient => "insufficient",
            Self::NotCertified => "not_certified",
            Self::Invalid => "invalid",
            Self::Unevaluated => "unevaluated",
        }
    }
}

/// The re-verified derivation behind a sufficient subset.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepairDerivation {
    /// The theorem-specific checker that decided.
    pub checker: &'static str,
    /// Stable facts of the derivation (stages searched, adjustment set, ...).
    pub steps: Vec<String>,
    /// Whether the independent re-verification passed.
    pub verified: bool,
}

/// A family's verdict on one hypothetical evidence set.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum FamilyVerdict {
    /// The checker identified the contract; the derivation is attached.
    Sufficient(RepairDerivation),
    /// The checker ran and the contract is still unmet.
    Insufficient {
        /// Why, in the checker's own words.
        reasons: Vec<String>,
    },
    /// The checker could not certify the evidence.
    NotCertified {
        /// Why.
        reasons: Vec<String>,
    },
    /// The hypothetical evidence is malformed.
    Invalid {
        /// Why.
        reasons: Vec<String>,
    },
}

/// Cooperative cancellation observed by a family's checker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RepairCancelled;

/// One failed causal contract and the theorem-specific checker that decides it.
pub trait RepairFamily {
    /// Stable family name (`transport`, `backdoor`, ...).
    fn family_id(&self) -> &'static str;

    /// Stable identity of the failed contract.
    fn contract_id(&self) -> String;

    /// The unresolved obligations of the failed contract, each retaining its
    /// source proof step.
    fn unresolved_obligations(&self) -> Vec<EvidenceObligation>;

    /// Re-run the theorem-specific identifier/validator with the evidence the
    /// `studies` would deliver, as a hypothesis.
    ///
    /// # Errors
    /// Cancellation, which stops the whole repair search.
    fn check(
        &self,
        studies: &[&DurableStudyCandidate],
        ctx: &ExecutionContext,
    ) -> Result<FamilyVerdict, RepairCancelled>;
}

/// How sufficient subsets are ranked.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RepairObjective {
    /// Least declared cost units, then sample budget, then semantic ids. All
    /// candidates must declare the same cost unit label.
    MinimizeCost,
    /// Least sample budget, then cost units, then semantic ids.
    MinimizeSampleBudget,
}

/// Declared search bounds: operations and depth (the subset size) under one
/// [`SearchBudget`], and its memory cap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RepairLimits {
    /// Operation and depth limits; depth is the largest subset size examined.
    pub search: SearchLimits,
    /// Memory cap of the budget, in bytes (the context's hard limit lowers it).
    pub memory_limit_bytes: u64,
}

impl Default for RepairLimits {
    fn default() -> Self {
        Self {
            search: SearchLimits { operations: 4096, depth: 3 },
            memory_limit_bytes: 64 * 1024 * 1024,
        }
    }
}

/// A refused repair request: a registered reason code and a stable detail.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct RepairError {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `identification_repair.*` detail.
    pub detail: &'static str,
    /// Human-readable explanation.
    pub message: String,
    /// The budget's receipt, for a stop before the search was entered.
    pub receipt: Option<Box<SearchReceipt>>,
}

impl RepairError {
    fn new(code: &'static str, detail: &'static str, message: impl Into<String>) -> Self {
        Self { code, detail, message: message.into(), receipt: None }
    }

    fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(
            reason_code!("invalid_argument"),
            "identification_repair.invalid_request",
            message,
        )
    }

    fn bounds(message: impl Into<String>) -> Self {
        Self::new(
            reason_code!("invalid_argument"),
            "identification_repair.bounds_exceeded",
            message,
        )
    }

    fn not_a_failure(message: impl Into<String>) -> Self {
        Self::new(reason_code!("invalid_argument"), "identification_repair.not_a_failure", message)
    }

    fn budget(message: impl Into<String>, receipt: Option<SearchReceipt>) -> Self {
        Self {
            code: reason_code!("transport_budget_cancel"),
            detail: "identification_repair.budget",
            message: message.into(),
            receipt: receipt.map(Box::new),
        }
    }

    fn from_identification(error: &IdentificationError) -> Self {
        if error.is_budget_or_cancel() {
            Self::budget(format!("the base identification stopped: {error}"), None)
        } else {
            Self::invalid_request(error.to_string())
        }
    }
}

impl RepairLimits {
    fn check_bounds(&self, candidates: usize) -> Result<(), RepairError> {
        if candidates > REPAIR_MAX_CANDIDATES
            || self.search.depth > REPAIR_MAX_DEPTH
            || self.search.operations > REPAIR_MAX_OPERATIONS
            || self.memory_limit_bytes > REPAIR_MAX_MEMORY_BYTES
        {
            return Err(RepairError::bounds(format!(
                "a repair request declares at most {REPAIR_MAX_CANDIDATES} candidates, subset \
                 depth {REPAIR_MAX_DEPTH}, {REPAIR_MAX_OPERATIONS} operations and \
                 {REPAIR_MAX_MEMORY_BYTES} bytes"
            )));
        }
        Ok(())
    }
}

/// One candidate or candidate subset and how it fared.
#[derive(Clone, Debug, PartialEq)]
pub struct RepairOutcome {
    /// Semantic candidate ids of the subset, sorted.
    pub candidates: Vec<Arc<str>>,
    /// Classification.
    pub classification: RepairClassification,
    /// Total declared cost units.
    pub cost_units: u64,
    /// Cost unit label shared by every member; `None` when members differ.
    pub cost_unit_label: Option<Arc<str>>,
    /// Total declared sample budget.
    pub sample_budget: u64,
    /// Failure or downgrade reasons, retained verbatim.
    pub reasons: Vec<String>,
    /// Obligation ids some member's expected evidence passes the necessary
    /// screen for ([`EvidenceObligation::addressed_by`]); a screen, not a verdict.
    pub addressed: Vec<Arc<str>>,
    /// Obligation ids no member addresses.
    pub unmet: Vec<Arc<str>>,
    /// The re-verified derivation, when the checker identified.
    pub derivation: Option<RepairDerivation>,
}

/// What the subset search consumed and left.
#[derive(Clone, Debug, PartialEq)]
pub struct RepairReceipt {
    /// Limits in force.
    pub limits: RepairLimits,
    /// Operations charged.
    pub operations_consumed: usize,
    /// Deepest subset size a charge was attempted at.
    pub depth_reached: usize,
    /// Effective memory cap in force.
    pub memory_limit_bytes: u64,
    /// Subsets evaluated, in order (at most [`REPAIR_RECEIPT_REGIONS`]).
    pub explored: Vec<String>,
    /// Subsets left unevaluated by a stop (at most [`REPAIR_RECEIPT_REGIONS`]).
    pub unevaluated: Vec<String>,
    /// Total unevaluated subsets, listed or not.
    pub unevaluated_total: usize,
    /// Supersets of a sufficient subset skipped: ranking only, never a verdict.
    pub dominated_skipped: usize,
    /// Whether subsets larger than the declared depth exist and were not
    /// examined; never a verdict about them.
    pub beyond_declared_depth: bool,
    /// The budget's receipt when operations, depth, memory or cancellation
    /// stopped the search.
    pub stop: Option<SearchReceipt>,
}

/// A finished repair search.
#[derive(Clone, Debug, PartialEq)]
pub struct RepairReport {
    /// The family that decided.
    pub family: &'static str,
    /// The failed contract's identity.
    pub contract: String,
    /// The ranking objective.
    pub objective: RepairObjective,
    /// The unresolved obligations the candidates were screened against.
    pub obligations: Vec<EvidenceObligation>,
    /// Invalid candidates (by semantic id), then every evaluated or
    /// unevaluated subset in search order.
    pub outcomes: Vec<RepairOutcome>,
    /// Indices into `outcomes` of verified-sufficient subsets, best first.
    pub ranked_sufficient: Vec<usize>,
    /// The search receipt.
    pub receipt: RepairReceipt,
}

impl RepairReport {
    /// The best verified-sufficient subset under the objective.
    #[must_use]
    pub fn best(&self) -> Option<&RepairOutcome> {
        self.ranked_sufficient.first().and_then(|i| self.outcomes.get(*i))
    }

    /// The `(reason code, detail)` of a report with no verified-sufficient
    /// subset: a budget stop, an unresolved assumption no study can satisfy, or
    /// no certified repair (`identification_repair.wrong_contract`, whose
    /// outcomes keep the failure reasons). `None` when a repair was verified.
    #[must_use]
    pub fn status(&self) -> Option<(&'static str, &'static str)> {
        if !self.ranked_sufficient.is_empty() {
            return None;
        }
        if self.receipt.stop.is_some() {
            return Some((reason_code!("transport_budget_cancel"), "identification_repair.budget"));
        }
        if self.obligations.iter().any(|o| !o.satisfiable_by_study()) {
            return Some((
                reason_code!("transport_missing_evidence"),
                "evidence_obligations.wrong_contract",
            ));
        }
        Some((reason_code!("transport_not_certified"), "identification_repair.wrong_contract"))
    }
}

fn subset_label(ids: &[Arc<str>]) -> String {
    format!("subset:[{}]", ids.iter().map(AsRef::as_ref).collect::<Vec<&str>>().join(","))
}

/// Lexicographic `k`-combinations of `0..n`.
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if k == 0 || k > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.clone());
        let mut i = k;
        let pos = loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if idx[i] != i + n - k {
                break i;
            }
        };
        idx[pos] += 1;
        let start = idx[pos];
        for (offset, slot) in idx.iter_mut().skip(pos + 1).enumerate() {
            *slot = start + 1 + offset;
        }
    }
}

fn subset_cost(members: &[&DurableStudyCandidate]) -> (u64, Option<Arc<str>>, u64) {
    let units = members.iter().fold(0_u64, |sum, m| sum.saturating_add(m.cost.units));
    let samples = members.iter().fold(0_u64, |sum, m| sum.saturating_add(m.cost.sample_budget));
    let label = members
        .first()
        .map(|m| Arc::clone(&m.cost.unit_label))
        .filter(|first| members.iter().all(|m| m.cost.unit_label.as_ref() == first.as_ref()));
    (units, label, samples)
}

/// Obligation ids the members' expected evidence passes the necessary screen
/// for, and the ids that no member addresses.
fn screen(
    obligations: &[EvidenceObligation],
    members: &[&DurableStudyCandidate],
) -> (Vec<Arc<str>>, Vec<Arc<str>>) {
    let offers: Vec<_> = members.iter().flat_map(|m| m.offers()).collect();
    let mut addressed = Vec::new();
    let mut unmet = Vec::new();
    for obligation in obligations {
        if offers.iter().any(|offer| obligation.addressed_by(offer)) {
            addressed.push(Arc::clone(&obligation.id));
        } else {
            unmet.push(Arc::clone(&obligation.id));
        }
    }
    (addressed, unmet)
}

fn classify(
    verdict: FamilyVerdict,
    blocking: &[Arc<str>],
) -> (RepairClassification, Vec<String>, Option<RepairDerivation>) {
    match verdict {
        FamilyVerdict::Sufficient(derivation) if !derivation.verified => (
            RepairClassification::NotCertified,
            vec!["the checker's derivation did not pass independent re-verification".to_owned()],
            Some(derivation),
        ),
        FamilyVerdict::Sufficient(derivation) if !blocking.is_empty() => (
            RepairClassification::NotCertified,
            vec![format!(
                "evidence_obligations.wrong_contract: a study does not establish assumption \
                 obligation(s) {}",
                blocking.iter().map(AsRef::as_ref).collect::<Vec<&str>>().join(", ")
            )],
            Some(derivation),
        ),
        FamilyVerdict::Sufficient(derivation) => {
            (RepairClassification::VerifiedSufficient, Vec::new(), Some(derivation))
        }
        FamilyVerdict::Insufficient { reasons } => {
            (RepairClassification::Insufficient, reasons, None)
        }
        FamilyVerdict::NotCertified { reasons } => {
            (RepairClassification::NotCertified, reasons, None)
        }
        FamilyVerdict::Invalid { reasons } => (RepairClassification::Invalid, reasons, None),
    }
}

struct Searcher<'a> {
    family: &'a dyn RepairFamily,
    studies: &'a [(Arc<str>, &'a DurableStudyCandidate)],
    obligations: &'a [EvidenceObligation],
    blocking: Vec<Arc<str>>,
}

impl Searcher<'_> {
    fn outcome(
        &self,
        combo: &[usize],
        classification: RepairClassification,
        reasons: Vec<String>,
        derivation: Option<RepairDerivation>,
    ) -> RepairOutcome {
        let members: Vec<&DurableStudyCandidate> =
            combo.iter().map(|i| self.studies[*i].1).collect();
        let (cost_units, cost_unit_label, sample_budget) = subset_cost(&members);
        let (addressed, unmet) = screen(self.obligations, &members);
        RepairOutcome {
            candidates: combo.iter().map(|i| Arc::clone(&self.studies[*i].0)).collect(),
            classification,
            cost_units,
            cost_unit_label,
            sample_budget,
            reasons,
            addressed,
            unmet,
            derivation,
        }
    }

    fn evaluate(
        &self,
        combo: &[usize],
        ctx: &ExecutionContext,
    ) -> Result<RepairOutcome, RepairCancelled> {
        let members: Vec<&DurableStudyCandidate> =
            combo.iter().map(|i| self.studies[*i].1).collect();
        let verdict = self.family.check(&members, ctx)?;
        let (classification, reasons, derivation) = classify(verdict, &self.blocking);
        Ok(self.outcome(combo, classification, reasons, derivation))
    }
}

/// The subsets of one search walk.
struct Walk {
    evaluated: Vec<RepairOutcome>,
    unevaluated: Vec<RepairOutcome>,
    unevaluated_total: usize,
    dominated: usize,
    stop: Option<SearchStop>,
}

/// Whether `combo` contains a subset already verified sufficient.
fn covered(sufficient: &[&Vec<usize>], combo: &[usize]) -> bool {
    sufficient.iter().any(|s| s.iter().all(|i| combo.contains(i)))
}

impl Searcher<'_> {
    /// Evaluate `combos` in order, charging one operation per subset at depth
    /// equal to its size. Supersets of a sufficient subset are skipped (ranking
    /// only); a stop leaves every remaining subset unevaluated.
    fn walk(
        &self,
        combos: &[Vec<usize>],
        budget: &mut SearchBudget<'_>,
        ctx: &ExecutionContext,
    ) -> Walk {
        let mut walk = Walk {
            evaluated: Vec::new(),
            unevaluated: Vec::new(),
            unevaluated_total: 0,
            dominated: 0,
            stop: None,
        };
        let mut sufficient: Vec<&Vec<usize>> = Vec::new();
        let mut stopped_at = None;
        for (at, combo) in combos.iter().enumerate() {
            if covered(&sufficient, combo) {
                walk.dominated += 1;
                continue;
            }
            let bytes = (combo.len() as u64).saturating_mul(1024).saturating_add(4096);
            if let Err(stop) = budget.charge(combo.len(), bytes) {
                walk.stop = Some(stop);
                stopped_at = Some(at);
                break;
            }
            match self.evaluate(combo, ctx) {
                Err(RepairCancelled) => {
                    walk.stop = Some(SearchStop::Cancelled);
                    stopped_at = Some(at);
                    break;
                }
                Ok(outcome) => {
                    if outcome.classification == RepairClassification::VerifiedSufficient {
                        sufficient.push(combo);
                    }
                    walk.evaluated.push(outcome);
                }
            }
        }
        if let Some(at) = stopped_at {
            for combo in &combos[at..] {
                if covered(&sufficient, combo) {
                    continue;
                }
                walk.unevaluated_total += 1;
                if walk.unevaluated.len() < REPAIR_RECEIPT_REGIONS {
                    walk.unevaluated.push(self.outcome(
                        combo,
                        RepairClassification::Unevaluated,
                        Vec::new(),
                        None,
                    ));
                }
            }
        }
        walk
    }
}

/// Validated, sorted, distinct candidates plus an outcome per refused one.
fn prepare<'a>(
    candidates: &'a [DurableStudyCandidate],
    obligations: &[EvidenceObligation],
) -> (Vec<(Arc<str>, &'a DurableStudyCandidate)>, Vec<RepairOutcome>) {
    let mut valid: Vec<(Arc<str>, &DurableStudyCandidate)> = Vec::new();
    let mut invalid: Vec<RepairOutcome> = Vec::new();
    let refuse = |id: Arc<str>, candidate: &DurableStudyCandidate, reason: String| RepairOutcome {
        candidates: vec![id],
        classification: RepairClassification::Invalid,
        cost_units: candidate.cost.units,
        cost_unit_label: Some(Arc::clone(&candidate.cost.unit_label)),
        sample_budget: candidate.cost.sample_budget,
        reasons: vec![reason],
        addressed: Vec::new(),
        unmet: obligations.iter().map(|o| Arc::clone(&o.id)).collect(),
        derivation: None,
    };
    for candidate in candidates {
        let id = candidate.semantic_id();
        match candidate.validate() {
            Err(error) => invalid.push(refuse(id, candidate, error.to_string())),
            Ok(()) if !candidate.feasible => invalid.push(refuse(
                id,
                candidate,
                "the candidate declares itself infeasible".to_owned(),
            )),
            Ok(()) => valid.push((id, candidate)),
        }
    }
    valid.sort_by(|a, b| a.0.cmp(&b.0));
    let mut distinct: Vec<(Arc<str>, &DurableStudyCandidate)> = Vec::with_capacity(valid.len());
    for (id, candidate) in valid {
        if distinct.last().is_some_and(|(last, _)| last == &id) {
            invalid.push(refuse(id, candidate, "duplicate semantic candidate id".to_owned()));
        } else {
            distinct.push((id, candidate));
        }
    }
    invalid.sort_by(|a, b| a.candidates.cmp(&b.candidates).then_with(|| a.reasons.cmp(&b.reasons)));
    (distinct, invalid)
}

fn rank(outcomes: &[RepairOutcome], objective: RepairObjective) -> Vec<usize> {
    let mut ranked: Vec<usize> = outcomes
        .iter()
        .enumerate()
        .filter(|(_, o)| o.classification == RepairClassification::VerifiedSufficient)
        .map(|(i, _)| i)
        .collect();
    ranked.sort_by(|&a, &b| {
        let (x, y) = (&outcomes[a], &outcomes[b]);
        let by_cost = x.cost_units.cmp(&y.cost_units);
        let by_samples = x.sample_budget.cmp(&y.sample_budget);
        match objective {
            RepairObjective::MinimizeCost => by_cost.then(by_samples),
            RepairObjective::MinimizeSampleBudget => by_samples.then(by_cost),
        }
        .then_with(|| x.candidates.cmp(&y.candidates))
    });
    ranked
}

/// Apply hypothetical study evidence to a failed contract and report which
/// candidates and bounded subsets repair it, under one declared budget.
///
/// Candidates are validated first (an invalid or infeasible candidate is an
/// `invalid` outcome and never searched); the rest are searched in semantic-id
/// order, by subset size, so the result does not depend on declaration order.
/// A subset is `verified_sufficient` only when the family's theorem checker
/// identifies the hypothetical evidence and its derivation re-verifies, and no
/// unresolved assumption obligation (never satisfiable by a study) remains.
///
/// # Errors
/// `identification_repair.bounds_exceeded` for a request beyond the declared
/// bounds, `identification_repair.cost_units_mismatch` when cost ranking
/// compares different unit labels, and `identification_repair.budget` when the
/// budget is stopped before the search is entered. A stop during the search is
/// a report, never an error.
pub fn repair(
    family: &dyn RepairFamily,
    candidates: &[DurableStudyCandidate],
    objective: RepairObjective,
    limits: RepairLimits,
    ctx: &ExecutionContext,
) -> Result<RepairReport, RepairError> {
    limits.check_bounds(candidates.len())?;
    let obligations = family.unresolved_obligations();
    let (studies, mut outcomes) = prepare(candidates, &obligations);
    if objective == RepairObjective::MinimizeCost
        && studies.iter().any(|(_, c)| c.cost.unit_label != studies[0].1.cost.unit_label)
    {
        return Err(RepairError::new(
            reason_code!("design_cost_units_mismatch"),
            "identification_repair.cost_units_mismatch",
            "cost ranking compares candidates that declare different cost unit labels",
        ));
    }
    let mut budget = SearchBudget::with_memory(limits.search, limits.memory_limit_bytes, ctx)
        .map_err(|receipt| {
            RepairError::budget("the repair budget stopped before the search", Some(receipt))
        })?;
    let searcher = Searcher {
        family,
        studies: &studies,
        obligations: &obligations,
        blocking: obligations
            .iter()
            .filter(|o| !o.satisfiable_by_study())
            .map(|o| Arc::clone(&o.id))
            .collect(),
    };
    let depth = limits.search.depth.min(studies.len());
    let combos: Vec<Vec<usize>> =
        (1..=depth).flat_map(|k| combinations(studies.len(), k)).collect();
    let walk = searcher.walk(&combos, &mut budget, ctx);
    let mut explored = Vec::new();
    for outcome in &walk.evaluated {
        if explored.len() < REPAIR_RECEIPT_REGIONS {
            explored.push(subset_label(&outcome.candidates));
        }
    }
    let mut unevaluated = Vec::new();
    for outcome in &walk.unevaluated {
        if unevaluated.len() < REPAIR_RECEIPT_REGIONS {
            unevaluated.push(subset_label(&outcome.candidates));
        }
    }
    let stop = walk.stop.map(|stop| budget.receipt(stop, explored.clone(), unevaluated.clone()));
    let unevaluated_total = walk.unevaluated_total;
    let dominated = walk.dominated;
    outcomes.extend(walk.evaluated);
    outcomes.extend(walk.unevaluated);
    let ranked_sufficient = rank(&outcomes, objective);
    Ok(RepairReport {
        family: family.family_id(),
        contract: family.contract_id(),
        objective,
        obligations,
        outcomes,
        ranked_sufficient,
        receipt: RepairReceipt {
            limits,
            operations_consumed: budget.operations(),
            depth_reached: budget.depth_reached(),
            memory_limit_bytes: budget.memory_limit_bytes(),
            explored,
            unevaluated,
            unevaluated_total,
            dominated_skipped: dominated,
            beyond_declared_depth: studies.len() > depth,
            stop,
        },
    })
}

/// The proposed regimes of every study in `studies` (a sample increase adds
/// none), with consecutive ids after the base catalog's largest.
fn hypothetical_regimes(
    base: &EvidenceCatalog,
    studies: &[&DurableStudyCandidate],
) -> Result<Vec<EvidenceRegime>, Vec<String>> {
    let mut next = base.regimes.iter().map(|r| r.id.raw()).max().map_or(0, |m| m.saturating_add(1));
    let mut regimes = Vec::new();
    for study in studies {
        let produced = study.proposed_regimes(next).map_err(|e| vec![e.to_string()])?;
        next = next.saturating_add(u32::try_from(produced.len()).unwrap_or(u32::MAX));
        regimes.extend(produced);
    }
    Ok(regimes)
}

/// The validated hypothetical delta of already-proposed `regimes`; `Ok(None)`
/// when there are none.
fn delta_of_regimes(
    base: &EvidenceCatalog,
    regimes: Vec<EvidenceRegime>,
) -> Result<Option<EvidenceCatalogDelta>, Vec<String>> {
    if regimes.is_empty() {
        return Ok(None);
    }
    if regimes
        .iter()
        .any(|r| !r.intervention_values.is_empty() && !valid_intervention_values(base, r))
    {
        return Err(vec![
            "a proposed intervention level lies outside the domain its population declares"
                .to_owned(),
        ]);
    }
    EvidenceCatalogDelta::try_new(base, regimes).map(Some).map_err(|e| vec![e.to_string()])
}

/// Classical (catalog-aware) transport as a repair family.
#[derive(Clone, Debug)]
pub struct TransportRepairFamily {
    diagram: SelectionDiagram,
    query: ClassicalTransportQuery,
    base: EvidenceCatalog,
    limits: SidLimits,
    obligations: Vec<EvidenceObligation>,
}

impl TransportRepairFamily {
    /// Freeze a failed transport contract: the base catalog must not already
    /// identify the query.
    ///
    /// # Errors
    /// `identification_repair.not_a_failure` when the base identifies,
    /// `identification_repair.invalid_request` for an invalid catalog or query,
    /// `identification_repair.budget` when the base search stops.
    pub fn try_new(
        diagram: SelectionDiagram,
        query: ClassicalTransportQuery,
        base: EvidenceCatalog,
        limits: SidLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, RepairError> {
        base.validate().map_err(|e| RepairError::invalid_request(e.to_string()))?;
        let result = identify_catalog_transport(&diagram, &query, &base, limits, ctx)
            .map_err(|e| RepairError::from_identification(&e))?;
        if matches!(result, CatalogTransportResult::Identified(_)) {
            return Err(RepairError::not_a_failure(
                "the base catalog already identifies the transport query",
            ));
        }
        let source = format!("transport:{}->{}", query.source, query.target);
        let obligations = transport_result_obligations(&result, &query, &base, &source)
            .map_err(|e| RepairError::invalid_request(e.to_string()))?;
        Ok(Self { diagram, query, base, limits, obligations })
    }

    fn verify(
        &self,
        bound: &BoundTransportFunctional,
        preview: &EvidenceCatalog,
        ctx: &ExecutionContext,
    ) -> Result<FamilyVerdict, RepairCancelled> {
        let proof = verify_classical_transport(
            &self.diagram,
            &self.query,
            bound.derivation(),
            self.limits,
            ctx,
        );
        if matches!(proof, Err(IdentificationError::Cancelled)) {
            return Err(RepairCancelled);
        }
        let rebound = bound.derivation().bind_catalog(preview);
        let mut steps: Vec<String> =
            bound.searched_alternatives().iter().map(|s| format!("stage:{s}")).collect();
        steps.push(format!("leaf_factors:{}", bound.leaf_factors().len()));
        Ok(FamilyVerdict::Sufficient(RepairDerivation {
            checker: "classical_transport.catalog",
            steps,
            verified: proof.is_ok() && rebound.is_ok(),
        }))
    }
}

impl RepairFamily for TransportRepairFamily {
    fn family_id(&self) -> &'static str {
        "transport"
    }

    fn contract_id(&self) -> String {
        let ids = |vs: &[VariableId]| {
            vs.iter().map(|v| v.raw().to_string()).collect::<Vec<_>>().join(",")
        };
        format!(
            "transport:{}->{}:outcomes[{}]:treatments[{}]",
            self.query.source,
            self.query.target,
            ids(&self.query.outcomes),
            ids(&self.query.treatments)
        )
    }

    fn unresolved_obligations(&self) -> Vec<EvidenceObligation> {
        self.obligations.clone()
    }

    fn check(
        &self,
        studies: &[&DurableStudyCandidate],
        ctx: &ExecutionContext,
    ) -> Result<FamilyVerdict, RepairCancelled> {
        match hypothetical_regimes(&self.base, studies) {
            Ok(regimes) => self.check_regimes(regimes, ctx),
            Err(reasons) => Ok(FamilyVerdict::Invalid { reasons }),
        }
    }
}

impl TransportRepairFamily {
    /// The failed contract's selection diagram.
    #[must_use]
    pub fn diagram(&self) -> &SelectionDiagram {
        &self.diagram
    }

    /// The failed contract's transport query.
    #[must_use]
    pub fn query(&self) -> &ClassicalTransportQuery {
        &self.query
    }

    /// The base evidence catalog the contract failed on.
    #[must_use]
    pub fn base_catalog(&self) -> &EvidenceCatalog {
        &self.base
    }

    /// The identification limits the checker runs under.
    #[must_use]
    pub const fn sid_limits(&self) -> SidLimits {
        self.limits
    }

    /// The proposed regimes `studies` would deliver on top of the base catalog.
    ///
    /// # Errors
    /// The failure reasons of an invalid study.
    pub fn hypothetical_regimes(
        &self,
        studies: &[&DurableStudyCandidate],
    ) -> Result<Vec<EvidenceRegime>, Vec<String>> {
        hypothetical_regimes(&self.base, studies)
    }

    /// Re-run the transport identifier and verifier on already-proposed
    /// `regimes` (the stored hypothetical delta of an artifact), exactly as
    /// [`RepairFamily::check`] does after it has built them from studies.
    ///
    /// # Errors
    /// Cancellation.
    pub fn check_regimes(
        &self,
        regimes: Vec<EvidenceRegime>,
        ctx: &ExecutionContext,
    ) -> Result<FamilyVerdict, RepairCancelled> {
        let delta = match delta_of_regimes(&self.base, regimes) {
            Ok(Some(delta)) => delta,
            Ok(None) => {
                return Ok(FamilyVerdict::NotCertified {
                    reasons: vec![
                        "no study of the subset delivers a regime; a sample increase has no \
                         transport checker"
                            .to_owned(),
                    ],
                });
            }
            Err(reasons) => return Ok(FamilyVerdict::Invalid { reasons }),
        };
        let preview = match delta.preview_catalog(&self.base) {
            Ok(preview) => preview,
            Err(error) => return Ok(FamilyVerdict::Invalid { reasons: vec![error.to_string()] }),
        };
        match identify_catalog_transport(&self.diagram, &self.query, &preview, self.limits, ctx) {
            Ok(CatalogTransportResult::Identified(bound)) => self.verify(&bound, &preview, ctx),
            Ok(CatalogTransportResult::MissingEvidence { obligations, .. }) => {
                Ok(FamilyVerdict::Insufficient {
                    reasons: obligations.iter().map(ToString::to_string).collect(),
                })
            }
            Ok(CatalogTransportResult::NotCertified { obligations, .. }) => {
                Ok(FamilyVerdict::NotCertified {
                    reasons: obligations.iter().map(ToString::to_string).collect(),
                })
            }
            Err(IdentificationError::Cancelled) => Err(RepairCancelled),
            Err(error) if error.is_budget_or_cancel() => Ok(FamilyVerdict::NotCertified {
                reasons: vec![format!("the bounded identification stopped: {error}")],
            }),
            Err(error) => Ok(FamilyVerdict::Invalid { reasons: vec![error.to_string()] }),
        }
    }
}

/// What the back-door identifier made of one joint law.
enum LawAnalysis {
    Identified { adjustment: Vec<VariableId> },
    NotIdentified(String),
    Failed(String),
}

/// Back-door adjustment as a repair family.
///
/// The failed contract is "identify the average effect of `treatment` on
/// `outcome` by back-door adjustment over the variables observed jointly with
/// them". A study repairs it only by delivering an observational, unconditional
/// joint law of the treatment, the outcome and covariates in the contract's
/// population: an adjustment set is read from one joint law, so covariates
/// measured in separate studies never combine.
#[derive(Clone, Debug)]
pub struct BackdoorRepairFamily {
    dag: Dag,
    treatment: VariableId,
    outcome: VariableId,
    population: Arc<str>,
    base_law: BTreeSet<VariableId>,
    obligations: Vec<EvidenceObligation>,
}

fn dense_of(dag: &Dag, variable: VariableId) -> Option<DenseNodeId> {
    dag.nodes()
        .iter()
        .position(|node| *node == NodeRef::Static(variable))
        .and_then(|i| DenseNodeId::try_from_usize(i).ok())
}

fn graph_variables(dag: &Dag) -> Vec<VariableId> {
    dag.nodes()
        .iter()
        .filter_map(|node| match node {
            NodeRef::Static(variable) => Some(*variable),
            _ => None,
        })
        .collect()
}

impl BackdoorRepairFamily {
    /// Freeze a failed back-door contract: the variables of `base_law` are those
    /// currently observed jointly (in `population`) with the treatment and
    /// outcome. `unresolved_assumptions` become `EstablishAssumption`
    /// obligations that no study can satisfy.
    ///
    /// # Errors
    /// `identification_repair.invalid_request` for an unknown or equal
    /// treatment/outcome or an empty population; `identification_repair.not_a_failure`
    /// when the base law already identifies the effect.
    pub fn try_new(
        dag: Dag,
        treatment: VariableId,
        outcome: VariableId,
        population: impl Into<Arc<str>>,
        base_law: impl IntoIterator<Item = VariableId>,
        unresolved_assumptions: &[ObligationRecord],
    ) -> Result<Self, RepairError> {
        let population = population.into();
        let variables = graph_variables(&dag);
        let base_law: BTreeSet<VariableId> = base_law.into_iter().collect();
        if treatment == outcome
            || !variables.contains(&treatment)
            || !variables.contains(&outcome)
            || base_law.iter().any(|v| !variables.contains(v))
            || population.trim().is_empty()
        {
            return Err(RepairError::invalid_request(
                "a back-door contract names a population and a treatment and outcome that differ \
                 and lie in the graph",
            ));
        }
        let mut family =
            Self { dag, treatment, outcome, population, base_law, obligations: Vec::new() };
        if family.law_usable(&family.base_law)
            && matches!(family.identify_in(&family.base_law), LawAnalysis::Identified { .. })
        {
            return Err(RepairError::not_a_failure(
                "the base law already identifies the effect by back-door adjustment",
            ));
        }
        family.obligations = family.derive_obligations(unresolved_assumptions)?;
        Ok(family)
    }

    fn law_usable(&self, law: &BTreeSet<VariableId>) -> bool {
        law.contains(&self.treatment) && law.contains(&self.outcome)
    }

    /// First admissible adjustment set (size order) over the whole graph, or
    /// `None` when none exists: the evidence the contract is waiting for.
    fn derive_obligations(
        &self,
        assumptions: &[ObligationRecord],
    ) -> Result<Vec<EvidenceObligation>, RepairError> {
        let everything: BTreeSet<VariableId> = graph_variables(&self.dag).into_iter().collect();
        let provenance = |step: String| ObligationProvenance {
            family: Arc::from("backdoor"),
            source: Arc::from(self.contract_id()),
            proof_step: Some(Arc::from(step)),
        };
        let mut out = Vec::new();
        match self.identify_in(&everything) {
            LawAnalysis::Identified { adjustment } => {
                let mut needed: BTreeSet<VariableId> = adjustment.iter().copied().collect();
                needed.insert(self.treatment);
                needed.insert(self.outcome);
                let list = needed.iter().map(|v| v.raw().to_string()).collect::<Vec<_>>().join(",");
                out.push(
                    EvidenceObligation::try_new(antecedent_core::EvidenceObligationSpec {
                        kind: EvidenceObligationKind::ProvideJointLaw,
                        scope: ObligationScope::Factor,
                        variables: needed.iter().copied().collect::<Vec<_>>().into(),
                        population: Some(Arc::clone(&self.population)),
                        regime: ObligationRegime::observational(true),
                        reason: Arc::from(format!(
                            "back-door adjustment needs one joint observational law over \
                             treatment, outcome and the adjustment set (first minimal set; \
                             variables {list})"
                        )),
                        required_slots: Arc::from([Arc::from("backdoor:adjustment_set")]),
                        min_additional_samples: None,
                        provenance: provenance(format!("backdoor.adjustment_set:{list}")),
                    })
                    .map_err(|e| RepairError::invalid_request(e.to_string()))?,
                );
            }
            LawAnalysis::NotIdentified(reason) | LawAnalysis::Failed(reason) => {
                out.push(
                    EvidenceObligation::try_new(antecedent_core::EvidenceObligationSpec {
                        kind: EvidenceObligationKind::EstablishAssumption,
                        scope: ObligationScope::Program,
                        variables: Arc::from([]),
                        population: None,
                        regime: ObligationRegime::observational(false),
                        reason: Arc::from(format!(
                            "no admissible back-door adjustment set exists in the graph: {reason}"
                        )),
                        required_slots: Arc::from([Arc::from("backdoor:no_admissible_set")]),
                        min_additional_samples: None,
                        provenance: provenance("backdoor.existence".to_owned()),
                    })
                    .map_err(|e| RepairError::invalid_request(e.to_string()))?,
                );
            }
        }
        out.extend(assumption_obligations(assumptions));
        Ok(out)
    }

    /// Run the back-door identifier over variables observed in `law`
    /// (everything else is forbidden) and independently verify its set.
    fn identify_in(&self, law: &BTreeSet<VariableId>) -> LawAnalysis {
        let forbidden: Vec<VariableId> =
            graph_variables(&self.dag).into_iter().filter(|v| !law.contains(v)).collect();
        let mut identifier = BackdoorIdentifier::new();
        identifier.config.forbidden = Arc::from(forbidden);
        let prepared = match identifier.prepare(&self.dag) {
            Ok(prepared) => prepared,
            Err(error) => return LawAnalysis::Failed(error.to_string()),
        };
        let query = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            self.treatment,
            self.outcome,
        ));
        let mut workspace = IdentificationWorkspace::default();
        match identifier.identify(&prepared, &query, &mut workspace) {
            Ok(result) if result.status == IdentificationStatus::NonparametricallyIdentified => {
                let Some(first) = result.estimands.first() else {
                    return LawAnalysis::Failed("identified result carries no estimand".to_owned());
                };
                let adjustment = first.adjustment_set.to_vec();
                if self.verify_adjustment(&adjustment, law) {
                    LawAnalysis::Identified { adjustment }
                } else {
                    LawAnalysis::Failed(
                        "the returned adjustment set failed independent verification".to_owned(),
                    )
                }
            }
            Ok(result) => LawAnalysis::NotIdentified(
                result
                    .derivation
                    .steps
                    .iter()
                    .map(|step| format!("{}: {}", step.rule, step.detail))
                    .collect::<Vec<_>>()
                    .join("; "),
            ),
            Err(error) => LawAnalysis::Failed(error.to_string()),
        }
    }

    /// The back-door criterion, re-checked without the identifier: `set` lies in
    /// the observed law, excludes treatment, outcome and every descendant of the
    /// treatment, and d-separates treatment from outcome once the treatment's
    /// outgoing edges are removed.
    fn verify_adjustment(&self, set: &[VariableId], law: &BTreeSet<VariableId>) -> bool {
        let (Some(t), Some(y)) =
            (dense_of(&self.dag, self.treatment), dense_of(&self.dag, self.outcome))
        else {
            return false;
        };
        let Some(z) = set.iter().map(|v| dense_of(&self.dag, *v)).collect::<Option<Vec<_>>>()
        else {
            return false;
        };
        if set.iter().any(|v| !law.contains(v) || *v == self.treatment || *v == self.outcome) {
            return false;
        }
        let mut descendants = BitSet::with_len(self.dag.node_count());
        self.dag.descendants_of(&[t], &mut descendants, &mut GraphWorkspace::default());
        if z.iter().any(|node| descendants.contains(*node)) {
            return false;
        }
        let mut mutilated = Dag::empty();
        for node in self.dag.nodes() {
            if mutilated.add_node(*node).is_err() {
                return false;
            }
        }
        for edge in self.dag.edges() {
            let Some((from, to)) = edge.parent_child() else { return false };
            if from != t && mutilated.insert_directed(from, to).is_err() {
                return false;
            }
        }
        mutilated.is_d_separated(t, y, &z, &mut DSeparationWorkspace::default()).unwrap_or(false)
    }
}

impl RepairFamily for BackdoorRepairFamily {
    fn family_id(&self) -> &'static str {
        "backdoor"
    }

    fn contract_id(&self) -> String {
        format!(
            "backdoor:{}:treatment={}:outcome={}",
            self.population,
            self.treatment.raw(),
            self.outcome.raw()
        )
    }

    fn unresolved_obligations(&self) -> Vec<EvidenceObligation> {
        self.obligations.clone()
    }

    fn check(
        &self,
        studies: &[&DurableStudyCandidate],
        ctx: &ExecutionContext,
    ) -> Result<FamilyVerdict, RepairCancelled> {
        if ctx.cancellation.is_cancelled() {
            return Err(RepairCancelled);
        }
        let mut laws: Vec<(String, BTreeSet<VariableId>)> = Vec::new();
        if self.law_usable(&self.base_law) {
            laws.push(("base".to_owned(), self.base_law.clone()));
        }
        let mut malformed = Vec::new();
        for study in studies {
            if let Err(error) = study.validate() {
                malformed.push(error.to_string());
                continue;
            }
            for (k, evidence) in study.expected_evidence.iter().enumerate() {
                let observational_joint = evidence.interventions.is_empty()
                    && evidence.conditioned_on.is_empty()
                    && matches!(
                        evidence.distribution,
                        antecedent_core::DistributionAvailability::Joint
                    );
                let law: BTreeSet<VariableId> = evidence.measured.iter().copied().collect();
                if evidence.population == self.population
                    && observational_joint
                    && self.law_usable(&law)
                {
                    laws.push((format!("{}#{k}", study.semantic_id()), law));
                }
            }
        }
        if !malformed.is_empty() {
            return Ok(FamilyVerdict::Invalid { reasons: malformed });
        }
        Ok(self.decide(&laws))
    }
}

impl BackdoorRepairFamily {
    /// The failed contract's graph.
    #[must_use]
    pub fn dag(&self) -> &Dag {
        &self.dag
    }

    /// The treatment variable.
    #[must_use]
    pub const fn treatment(&self) -> VariableId {
        self.treatment
    }

    /// The outcome variable.
    #[must_use]
    pub const fn outcome(&self) -> VariableId {
        self.outcome
    }

    /// The contract's population.
    #[must_use]
    pub fn population(&self) -> &str {
        &self.population
    }

    /// The variables currently observed jointly with treatment and outcome.
    #[must_use]
    pub fn base_law(&self) -> Vec<VariableId> {
        self.base_law.iter().copied().collect()
    }

    /// The proposed regimes `studies` would deliver, with consecutive ids from
    /// zero (a back-door contract has no base catalog).
    ///
    /// # Errors
    /// The failure reasons of an invalid study.
    pub fn hypothetical_regimes(
        &self,
        studies: &[&DurableStudyCandidate],
    ) -> Result<Vec<EvidenceRegime>, Vec<String>> {
        let mut next = 0_u32;
        let mut regimes = Vec::new();
        for study in studies {
            let produced = study.proposed_regimes(next).map_err(|e| vec![e.to_string()])?;
            next = next.saturating_add(u32::try_from(produced.len()).unwrap_or(u32::MAX));
            regimes.extend(produced);
        }
        Ok(regimes)
    }

    /// Re-run the back-door identifier on already-proposed `regimes` (the stored
    /// hypothetical delta of an artifact): only an observational, unconditional
    /// joint law in the contract's population that covers treatment and outcome
    /// counts, exactly as [`RepairFamily::check`] reads a study's expected
    /// evidence.
    ///
    /// # Errors
    /// Cancellation.
    pub fn check_regimes(
        &self,
        regimes: &[EvidenceRegime],
        ctx: &ExecutionContext,
    ) -> Result<FamilyVerdict, RepairCancelled> {
        if ctx.cancellation.is_cancelled() {
            return Err(RepairCancelled);
        }
        let mut laws: Vec<(String, BTreeSet<VariableId>)> = Vec::new();
        if self.law_usable(&self.base_law) {
            laws.push(("base".to_owned(), self.base_law.clone()));
        }
        for regime in regimes {
            let observational_joint = regime.interventions.is_empty()
                && regime.conditioned_on.is_empty()
                && matches!(regime.distribution, antecedent_core::DistributionAvailability::Joint);
            let law: BTreeSet<VariableId> = regime.measured.iter().copied().collect();
            if regime.population == self.population && observational_joint && self.law_usable(&law)
            {
                laws.push((format!("regime:{}", regime.id.raw()), law));
            }
        }
        Ok(self.decide(&laws))
    }

    /// The verdict over the usable joint laws, first identifying law wins.
    fn decide(&self, laws: &[(String, BTreeSet<VariableId>)]) -> FamilyVerdict {
        let mut reasons = Vec::new();
        for (name, law) in laws {
            match self.identify_in(law) {
                LawAnalysis::Identified { adjustment } => {
                    let set = adjustment
                        .iter()
                        .map(|v| v.raw().to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    return FamilyVerdict::Sufficient(RepairDerivation {
                        checker: "backdoor.adjustment",
                        steps: vec![format!("law:{name}"), format!("adjustment_set:[{set}]")],
                        verified: true,
                    });
                }
                LawAnalysis::NotIdentified(why) => reasons.push(format!("law {name}: {why}")),
                LawAnalysis::Failed(why) => {
                    return FamilyVerdict::NotCertified {
                        reasons: vec![format!("law {name}: {why}")],
                    };
                }
            }
        }
        if laws.is_empty() {
            reasons.push(
                "no observational joint law of the treatment and outcome in the contract's \
                 population"
                    .to_owned(),
            );
        }
        FamilyVerdict::Insufficient { reasons }
    }
}

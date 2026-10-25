//! Evidence plans: the options that would close an identification gap.
//!
//! A failed identification can often be answered by several kinds of evidence at
//! once: measure the adjustment set, measure the front-door mediators, find an
//! instrument, run an experiment on the hedge, measure the confounder a sensitivity
//! analysis names. Each is a theorem-specific route with its own
//! [`RepairFamily`]. An [`EvidencePlan`] collects what [`crate::repair()`] found on
//! every route into one ordered list of options, each stating
//!
//! * the **gate**: whether the route's own theorem checker verified the study
//!   (`verified_sufficient`), refused to certify it with a reason
//!   (`not_certified`: an assumption no study establishes, a bounded search),
//!   found it insufficient, left it unexamined by a budget (`unevaluated`) or found
//!   the declaration invalid;
//! * what it **resolves**: the obligations the study's evidence passes the
//!   necessary screen for, and what **remains unresolved**: every other
//!   obligation, including the assumptions a study can never establish;
//! * its declared **cost** in one unit, and, when the caller supplies a value
//!   evaluation, its **expected value of sample information** and the value net of
//!   the declared cost map.
//!
//! Options are ordered by the identification gate first and only then by value: a
//! study never outranks a verified one because it promises more information. Value
//! is the caller's declared decision, prior and signal evaluated by the existing
//! preposterior path; a subset of several studies has no value of its own (the
//! values of its members do not add), so it is listed unvalued and ordered after
//! the valued options of its gate. Costs are compared only in equal units. A search
//! a budget stopped reports the budget, never a scientific impossibility, and the
//! ordering carries no coverage claim: it is `point_only` with calibration
//! `unmeasured`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_contract::{EvidenceObligation, EvidenceObligationKind};
use antecedent_core::{CanonicalWriter, reason_code};

use crate::evsi::CostToUtilityMap;
use crate::ranking::{DesignRankingBasis, EvsiRankRow, RankingRecord};
use crate::repair::{RepairClassification, RepairDerivation, RepairReport};
use crate::study_candidate::DurableStudyCandidate;

/// The ranking guarantee of every plan: a point ordering, no rank-error coverage.
pub const PLAN_RANKING_GUARANTEE: &str = "point_only";
/// Calibration of every plan: no coverage was measured.
pub const PLAN_CALIBRATION: &str = "unmeasured";
/// What a stopped search does and does not say.
pub const PLAN_BUDGET_STATEMENT: &str = "the search stopped on a declared budget; subsets left \
unexamined are unknown, not impossible, and no scientific conclusion follows from the stop";
/// Why a subset has no value of its own.
pub const SUBSET_VALUE_NOTE: &str =
    "the value of a subset of studies is not the sum of its members' values, so it is not valued";

/// A refused plan: a registered reason code and a stable detail.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct EvidencePlanError {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `evidence_plan.*` detail.
    pub detail: &'static str,
    /// Human-readable explanation.
    pub message: String,
}

impl EvidencePlanError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("invalid_argument"),
            detail: "evidence_plan.invalid_request",
            message: message.into(),
        }
    }

    fn units(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("design_cost_units_mismatch"),
            detail: "evidence_plan.cost_units_mismatch",
            message: message.into(),
        }
    }
}

/// A route no theorem checker decides, with the obligations it leaves open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnverifiedRoute {
    /// The route (obligation provenance family).
    pub route: Arc<str>,
    /// The obligations that route owes.
    pub obligations: Vec<Arc<str>>,
    /// Why nothing verifies it.
    pub reason: Arc<str>,
}

/// What a route's search consumed and left.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteBudget {
    /// Subsets left unexamined by a stop.
    pub unevaluated_total: usize,
    /// The stop that ended the search (`search.operations`, ...), when one did.
    pub stop: Option<String>,
    /// Whether subsets larger than the declared depth exist and were not examined.
    pub beyond_declared_depth: bool,
}

/// One theorem route the plan covers.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanRoute {
    /// The family that decided (`backdoor`, `frontdoor`, `instrument`, ...).
    pub family: &'static str,
    /// The failed contract's identity.
    pub contract: String,
    /// The obligations the route's candidates were screened against.
    pub obligations: Vec<EvidenceObligation>,
    /// What the search consumed and left.
    pub budget: RouteBudget,
}

/// Declared cost of an option, in one unit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanCost {
    /// Total declared cost units.
    pub units: u64,
    /// The unit label shared by every study of the plan.
    pub unit_label: Arc<str>,
    /// Total declared sample budget.
    pub sample_budget: u64,
}

/// The value of a single study under the declared decision, prior and signal.
#[derive(Clone, Debug, PartialEq)]
pub struct OptionValue {
    /// Expected value of sample information.
    pub evsi: f64,
    /// Expected value of perfect information (an upper bound on `evsi`).
    pub evpi: f64,
    /// Standard error of the EVSI (zero unless Monte Carlo).
    pub mc_stderr: f64,
    /// `exact`, `monte_carlo` or `externally_computed`.
    pub integration: &'static str,
    /// The study's cost in utility units, only with a cost map.
    pub study_cost_utility: Option<f64>,
    /// `evsi` minus the cost in utility units, only with a cost map.
    pub net_value: Option<f64>,
    /// Provider trust label of the signal.
    pub provider_trust: &'static str,
}

/// One option: a study or bounded subset of studies on one route.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanOption {
    /// Position in the plan's order (0 first).
    pub rank: usize,
    /// The route that decided.
    pub route: &'static str,
    /// Semantic ids of the studies, sorted.
    pub members: Vec<Arc<str>>,
    /// Their human labels.
    pub labels: Vec<Arc<str>>,
    /// The route's verdict on the option.
    pub gate: RepairClassification,
    /// The checker's reasons, verbatim (why it was not certified, insufficient or
    /// invalid).
    pub reasons: Vec<String>,
    /// The re-verified derivation, when the gate is `verified_sufficient` or the
    /// checker identified but an assumption blocks certification.
    pub derivation: Option<RepairDerivation>,
    /// Obligation ids the studies' evidence passes the necessary screen for.
    pub resolves: Vec<Arc<str>>,
    /// Every other obligation of the route.
    pub unresolved: Vec<Arc<str>>,
    /// The unresolved obligations no study can establish (assumptions).
    pub assumptions_remaining: Vec<Arc<str>>,
    /// Declared cost.
    pub cost: PlanCost,
    /// Value, for a single study with a value evaluation.
    pub value: Option<OptionValue>,
    /// Why the option is unvalued, when it is.
    pub value_note: Option<&'static str>,
    /// The gap to the next option of the same gate is within the value's error.
    pub rank_uncertain: bool,
}

/// What a plan can say about closing the gap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanOutcome {
    /// At least one option is verified sufficient.
    Closing,
    /// Nothing declared was certified; no impossibility follows.
    NoneCertified,
    /// A budget stopped a search with nothing certified; no verdict follows.
    BudgetExhausted,
}

impl PlanOutcome {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Closing => "closing",
            Self::NoneCertified => "none_certified",
            Self::BudgetExhausted => "budget_exhausted",
        }
    }
}

/// What the options are valued by.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanValueBasis {
    /// No value evaluation: gate, then cost.
    None,
    /// Expected value of sample information (no cost map).
    Evsi,
    /// EVSI net of the declared cost map.
    NetValue,
}

impl PlanValueBasis {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Evsi => "evsi",
            Self::NetValue => "net_value",
        }
    }
}

/// The ordered options that close an identification gap.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidencePlan {
    /// The theorem routes the plan covers.
    pub routes: Vec<PlanRoute>,
    /// Routes no theorem checker decides.
    pub unverified: Vec<UnverifiedRoute>,
    /// The options, best first.
    pub options: Vec<PlanOption>,
    /// What the plan can say.
    pub outcome: PlanOutcome,
    /// The one cost unit every option is compared in.
    pub cost_unit: Arc<str>,
    /// What the options are valued by.
    pub basis: PlanValueBasis,
    /// The declared cost-to-utility mapping, when one was used.
    pub cost_map: Option<CostToUtilityMap>,
    /// Utility unit of the value, when valued.
    pub utility_unit: Option<String>,
    /// Identity of the value evaluation the values came from.
    pub value_identity: Option<String>,
}

/// One study's value row.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanValueRow {
    /// The study's semantic id.
    pub semantic_id: String,
    /// Its value.
    pub value: OptionValue,
    /// The value ranking could not separate it from a neighbour.
    pub rank_uncertain: bool,
}

/// The value evaluation a plan is valued by: the studies that carry a signal
/// declaration, valued under one decision, prior and cost map.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanValue {
    /// What the values are ordered by.
    pub basis: PlanValueBasis,
    /// The declared cost-to-utility mapping, when one was used.
    pub cost_map: Option<CostToUtilityMap>,
    /// Utility unit of the decision.
    pub utility_unit: String,
    /// Identity of the evaluation (the design-ranking artifact digest).
    pub identity: String,
    /// One row per valued study.
    pub rows: Vec<PlanValueRow>,
}

impl PlanValue {
    /// Read the values off a design ranking record (freshly evaluated or replayed
    /// from its artifact), bound to the ranking's `identity`.
    ///
    /// # Errors
    /// `evidence_plan.invalid_request` for a structural ranking, which carries no
    /// value.
    pub fn from_ranking(record: &RankingRecord, identity: &str) -> Result<Self, EvidencePlanError> {
        let basis = match record.basis {
            DesignRankingBasis::Evsi => PlanValueBasis::Evsi,
            DesignRankingBasis::NetValue => PlanValueBasis::NetValue,
            DesignRankingBasis::StructuralSufficiencyCost => {
                return Err(EvidencePlanError::invalid(
                    "a structural ranking carries no value to order a plan by",
                ));
            }
        };
        let utility_unit = record
            .utility_unit
            .clone()
            .ok_or_else(|| EvidencePlanError::invalid("the ranking names no utility unit"))?;
        let rows = record
            .entries
            .iter()
            .map(|entry| PlanValueRow {
                semantic_id: entry.semantic_id.clone(),
                value: value_of(entry, record.cost_mapping.as_ref()),
                rank_uncertain: entry.rank_uncertain,
            })
            .collect();
        Ok(Self {
            basis,
            cost_map: record.cost_mapping.clone(),
            utility_unit,
            identity: identity.to_owned(),
            rows,
        })
    }
}

/// The reports and declarations a plan is built from.
#[derive(Clone, Copy)]
pub struct PlanInputs<'a> {
    /// One repair report per route, over the same candidates.
    pub reports: &'a [RepairReport],
    /// Every declared candidate, valid or not.
    pub candidates: &'a [DurableStudyCandidate],
    /// The value evaluation of the studies that carry a signal declaration.
    pub value: Option<&'a PlanValue>,
}

const fn gate_rank(gate: RepairClassification) -> u8 {
    match gate {
        RepairClassification::VerifiedSufficient => 0,
        RepairClassification::NotCertified => 1,
        RepairClassification::Insufficient => 2,
        RepairClassification::Unevaluated => 3,
        _ => 4,
    }
}

fn value_of(row: &EvsiRankRow, map: Option<&CostToUtilityMap>) -> OptionValue {
    OptionValue {
        evsi: row.evsi,
        evpi: row.evpi,
        mc_stderr: row.mc_stderr,
        integration: row.integration.as_str(),
        study_cost_utility: map
            .filter(|_| row.net_value.is_some())
            .map(|m| m.utility_per_cost * row.study_cost_amount),
        net_value: row.net_value,
        provider_trust: row.provider_trust.as_str(),
    }
}

impl OptionValue {
    /// The number options are ordered by under `basis`.
    fn score(&self, basis: PlanValueBasis) -> Option<f64> {
        match basis {
            PlanValueBasis::None => None,
            PlanValueBasis::Evsi => Some(self.evsi),
            PlanValueBasis::NetValue => self.net_value,
        }
    }
}

fn compare(a: &PlanOption, b: &PlanOption, basis: PlanValueBasis) -> Ordering {
    let score = |o: &PlanOption| o.value.as_ref().and_then(|v| v.score(basis));
    gate_rank(a.gate)
        .cmp(&gate_rank(b.gate))
        .then_with(|| match (score(a), score(b)) {
            (Some(x), Some(y)) => y.total_cmp(&x),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        })
        .then(a.cost.units.cmp(&b.cost.units))
        .then(a.cost.sample_budget.cmp(&b.cost.sample_budget))
        .then_with(|| a.members.cmp(&b.members))
        .then_with(|| a.route.cmp(b.route))
}

/// Apply the declared gate/value ordering, then classify the complete plan.
fn rank_options(
    options: &mut [PlanOption],
    routes: &[PlanRoute],
    basis: PlanValueBasis,
) -> PlanOutcome {
    options.sort_by(|a, b| compare(a, b, basis));
    for (rank, option) in options.iter_mut().enumerate() {
        option.rank = rank;
    }
    mark_close_calls(options, basis);
    if options.iter().any(|o| o.gate == RepairClassification::VerifiedSufficient) {
        PlanOutcome::Closing
    } else if routes.iter().any(|r| r.budget.stop.is_some()) {
        PlanOutcome::BudgetExhausted
    } else {
        PlanOutcome::NoneCertified
    }
}

/// Validated common units and optional valuation used by every route option.
struct PlanValuation {
    cost_unit: Arc<str>,
    basis: PlanValueBasis,
    cost_map: Option<CostToUtilityMap>,
    utility_unit: Option<String>,
    value_identity: Option<String>,
}

impl PlanValuation {
    fn of(inputs: &PlanInputs<'_>) -> Result<Self, EvidencePlanError> {
        if inputs.reports.is_empty() {
            return Err(EvidencePlanError::invalid("a plan needs at least one route report"));
        }
        let units: BTreeSet<&str> =
            inputs.candidates.iter().map(|c| c.cost.unit_label.as_ref()).collect();
        let cost_unit = match units.len() {
            0 => return Err(EvidencePlanError::invalid("a plan needs declared studies")),
            1 => Arc::clone(&inputs.candidates[0].cost.unit_label),
            _ => {
                return Err(EvidencePlanError::units(format!(
                    "options are compared in one cost unit; the declared studies use {}",
                    units.iter().map(|u| format!("`{u}`")).collect::<Vec<_>>().join(", ")
                )));
            }
        };
        let (basis, cost_map, utility_unit, value_identity) = match inputs.value {
            None => (PlanValueBasis::None, None, None, None),
            Some(value) => {
                if let Some(map) = &value.cost_map {
                    if map.cost_unit.as_str() != cost_unit.as_ref() {
                        return Err(EvidencePlanError::units(format!(
                            "the cost map converts `{}` but the declared studies cost `{cost_unit}`",
                            map.cost_unit
                        )));
                    }
                }
                (
                    value.basis,
                    value.cost_map.clone(),
                    Some(value.utility_unit.clone()),
                    Some(value.identity.clone()),
                )
            }
        };
        Ok(Self { cost_unit, basis, cost_map, utility_unit, value_identity })
    }
}

impl EvidencePlan {
    /// Build the plan from per-route repair reports.
    ///
    /// # Errors
    /// `evidence_plan.invalid_request` for no reports, or a report that names a
    /// study that was not declared; `evidence_plan.cost_units_mismatch` (reason
    /// `design_cost_units_mismatch`) when the declared studies do not share one cost
    /// unit, or the cost map's cost unit is not that unit.
    pub fn build(
        inputs: PlanInputs<'_>,
        unverified: Vec<UnverifiedRoute>,
    ) -> Result<Self, EvidencePlanError> {
        let PlanValuation { cost_unit, basis, cost_map, utility_unit, value_identity } =
            PlanValuation::of(&inputs)?;
        let label_of = |id: &Arc<str>| -> Arc<str> {
            inputs
                .candidates
                .iter()
                .find(|c| c.semantic_id() == *id)
                .map_or_else(|| Arc::clone(id), |c| Arc::clone(&c.label))
        };
        let mut routes = Vec::with_capacity(inputs.reports.len());
        let mut options = Vec::new();
        // A malformed declaration is invalid on every route; list it once.
        let mut invalid_seen: BTreeSet<Vec<Arc<str>>> = BTreeSet::new();
        for report in inputs.reports {
            let assumptions: BTreeSet<&Arc<str>> = report
                .obligations
                .iter()
                .filter(|o| o.kind == EvidenceObligationKind::EstablishAssumption)
                .map(|o| &o.id)
                .collect();
            for outcome in &report.outcomes {
                let singleton = outcome.candidates.len() == 1;
                if (outcome.classification == RepairClassification::Insufficient
                    || outcome.classification == RepairClassification::Unevaluated)
                    && !singleton
                {
                    continue;
                }
                if outcome.classification == RepairClassification::Invalid
                    && !invalid_seen.insert(outcome.candidates.clone())
                {
                    continue;
                }
                if let Some(unknown) = outcome
                    .candidates
                    .iter()
                    .find(|id| !inputs.candidates.iter().any(|c| c.semantic_id() == **id))
                {
                    return Err(EvidencePlanError::invalid(format!(
                        "the report names study `{unknown}` that was not declared"
                    )));
                }
                if let Some(label) = &outcome.cost_unit_label {
                    if label.as_ref() != cost_unit.as_ref() {
                        return Err(EvidencePlanError::units(format!(
                            "an option costs `{label}` but the plan compares `{cost_unit}`"
                        )));
                    }
                }
                let row = inputs.value.filter(|_| singleton).and_then(|value| {
                    value.rows.iter().find(|r| r.semantic_id == *outcome.candidates[0])
                });
                options.push(PlanOption {
                    rank: 0,
                    route: report.family,
                    members: outcome.candidates.clone(),
                    labels: outcome.candidates.iter().map(&label_of).collect(),
                    gate: outcome.classification,
                    reasons: outcome.reasons.clone(),
                    derivation: outcome.derivation.clone(),
                    resolves: outcome.addressed.clone(),
                    unresolved: outcome.unmet.clone(),
                    assumptions_remaining: outcome
                        .unmet
                        .iter()
                        .filter(|id| assumptions.contains(id))
                        .cloned()
                        .collect(),
                    cost: PlanCost {
                        units: outcome.cost_units,
                        unit_label: Arc::clone(&cost_unit),
                        sample_budget: outcome.sample_budget,
                    },
                    value: row.map(|r| r.value.clone()),
                    value_note: (!singleton && inputs.value.is_some()).then_some(SUBSET_VALUE_NOTE),
                    rank_uncertain: row.is_some_and(|r| r.rank_uncertain),
                });
            }
            routes.push(PlanRoute {
                family: report.family,
                contract: report.contract.clone(),
                obligations: report.obligations.clone(),
                budget: RouteBudget {
                    unevaluated_total: report.receipt.unevaluated_total,
                    stop: report.receipt.stop.as_ref().map(|s| s.stop.code().to_owned()),
                    beyond_declared_depth: report.receipt.beyond_declared_depth,
                },
            });
        }
        let outcome = rank_options(&mut options, &routes, basis);
        Ok(Self {
            routes,
            unverified,
            options,
            outcome,
            cost_unit,
            basis,
            cost_map,
            utility_unit,
            value_identity,
        })
    }

    /// The `(reason code, detail)` of a plan with no verified option: a budget
    /// stop (never an impossibility), obligations only an assumption can meet, or
    /// no certified option within the declared studies. `None` when a verified
    /// option exists.
    #[must_use]
    pub fn status(&self) -> Option<(&'static str, &'static str)> {
        match self.outcome {
            PlanOutcome::Closing => None,
            PlanOutcome::BudgetExhausted => {
                Some((reason_code!("transport_budget_cancel"), "evidence_plan.budget"))
            }
            PlanOutcome::NoneCertified => {
                if self.options.iter().any(|o| {
                    o.gate == RepairClassification::NotCertified
                        && !o.assumptions_remaining.is_empty()
                }) {
                    Some((
                        reason_code!("transport_missing_evidence"),
                        "evidence_obligations.wrong_contract",
                    ))
                } else {
                    Some((reason_code!("transport_not_certified"), "evidence_plan.none_certified"))
                }
            }
        }
    }

    /// The ranking guarantee: a point ordering, no rank-error coverage.
    #[must_use]
    pub const fn ranking_guarantee(&self) -> &'static str {
        PLAN_RANKING_GUARANTEE
    }

    /// The calibration of the ordering: no coverage was measured.
    #[must_use]
    pub const fn calibration(&self) -> &'static str {
        PLAN_CALIBRATION
    }

    /// The options a verified route found sufficient, best first.
    pub fn closing(&self) -> impl Iterator<Item = &PlanOption> {
        self.options.iter().filter(|o| o.gate == RepairClassification::VerifiedSufficient)
    }

    /// The obligation with `id` on any route.
    #[must_use]
    pub fn obligation(&self, id: &str) -> Option<&EvidenceObligation> {
        self.routes.iter().flat_map(|r| r.obligations.iter()).find(|o| o.id.as_ref() == id)
    }

    /// Order-independent identity of the plan (BLAKE3 hex).
    #[must_use]
    pub fn identity(&self) -> String {
        let mut writer = CanonicalWriter::new("antecedent.evidence_plan.v1");
        writer
            .text(self.outcome.as_str())
            .text(self.basis.as_str())
            .text(&self.cost_unit)
            .opt_text(self.utility_unit.as_deref())
            .opt_text(self.value_identity.as_deref());
        match &self.cost_map {
            Some(map) => writer
                .word(1)
                .text(&map.cost_unit)
                .text(&map.utility_unit)
                .real(map.utility_per_cost),
            None => writer.word(0),
        };
        writer.word(self.routes.len() as u64);
        for route in &self.routes {
            writer
                .text(route.family)
                .text(&route.contract)
                .word(route.obligations.len() as u64)
                .word(route.budget.unevaluated_total as u64)
                .opt_text(route.budget.stop.as_deref())
                .word(u64::from(route.budget.beyond_declared_depth));
            for obligation in &route.obligations {
                writer.text(&obligation.id);
            }
        }
        writer.word(self.unverified.len() as u64);
        for route in &self.unverified {
            writer.text(&route.route).text(&route.reason);
            for id in &route.obligations {
                writer.text(id);
            }
        }
        writer.word(self.options.len() as u64);
        for option in &self.options {
            writer
                .word(option.rank as u64)
                .text(option.route)
                .text(option.gate.as_str())
                .word(option.cost.units)
                .word(option.cost.sample_budget)
                .word(u64::from(option.rank_uncertain))
                .word(option.members.len() as u64);
            for member in &option.members {
                writer.text(member);
            }
            for id in option.resolves.iter().chain(option.unresolved.iter()) {
                writer.text(id);
            }
            writer.word(option.reasons.len() as u64);
            for reason in &option.reasons {
                writer.text(reason);
            }
            match &option.value {
                Some(value) => writer
                    .word(1)
                    .real(value.evsi)
                    .real(value.evpi)
                    .real(value.mc_stderr)
                    .text(value.integration)
                    .opt_real(value.net_value),
                None => writer.word(0),
            };
        }
        writer.finish_hex()
    }
}

/// Flag an option whose value is within error of the next option of its gate.
fn mark_close_calls(options: &mut [PlanOption], basis: PlanValueBasis) {
    let scored: Vec<Option<(f64, f64)>> = options
        .iter()
        .map(|o| o.value.as_ref().and_then(|v| v.score(basis).map(|score| (score, v.mc_stderr))))
        .collect();
    for i in 0..options.len().saturating_sub(1) {
        if options[i].gate != options[i + 1].gate {
            continue;
        }
        if let (Some((a, ea)), Some((b, eb))) = (scored[i], scored[i + 1]) {
            if (a - b).abs() <= (ea + eb).max(1e-12) {
                options[i].rank_uncertain = true;
            }
        }
    }
}

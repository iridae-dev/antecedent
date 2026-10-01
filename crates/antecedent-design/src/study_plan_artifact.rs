//! Durable study-plan artifact (`study_plan_v1`) and its independent consumer.
//!
//! The artifact stores the plan's premises (route, frozen rule-set and ranking
//! versions, graph, canonical query, every candidate declaration and the plan
//! limits), its data lineage (the whole base catalog: environments, regimes and
//! every binding's snapshot), and the plan outcome (frozen base failure, every
//! subset's outcome, ranked proposals with their repaired factors and required
//! margins, the stop receipt and the minimality flag). A premises digest, a
//! data digest and a plan digest over both seal it.
//!
//! The consumer refuses stored limits above its own maxima before any work,
//! recomputes every digest, then replays the whole plan under the stored
//! limits and accepts only an identical outcome. Replay protects against any
//! edit of the outcome and against premises or data that do not produce the
//! stored outcome; it does NOT protect against a producer that states a
//! different base catalog or candidate universe and re-seals honestly (the
//! artifact is then a correct plan of those stated inputs), and it does not
//! check that the arriving studies will deliver the declared regimes.
//!
//! Lineage is stored and digested, but replay does not exercise it: the plan
//! reads the base catalog's regimes, not its bindings, so a re-sealed artifact
//! with every base binding (snapshot) cleared replays to the same plan and is
//! accepted. Only [`StudyPlanProposal::receive`](crate::StudyPlanProposal::receive)
//! checks lineage: the arriving catalog must keep every stored base binding.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    ExecutionContext, IdentityDomain, InterventionAssignment, RegimeId, SearchLimits, SearchStop,
    VariableId, reason_code,
};
use antecedent_identify::{
    MIXED_SOURCE_RULE_SET, MixedSourceDerivationRecord, MixedSourceQuery,
    MzTransportDerivationRecord, MzTransportQuery, STUDY_PLAN_DEFAULT_LIMITS,
    STUDY_PLAN_MEMORY_BYTES, STUDY_PLAN_RANKING, StudyPlan, StudyPlanLimits, StudyPlanRoute,
    StudyPlanStop, StudyProposalDerivation, StudySubsetOutcome, ZTransportSourceSpec,
};
use antecedent_io::mixed_source_artifact::MixedSourceQueryWire;
use antecedent_io::mz_transport_artifact::MzSourceWire;
use antecedent_io::query_wire::ValueWire;
use antecedent_io::transport_catalog_wire::EvidenceCatalogWire;
use serde::{Deserialize, Serialize};

use crate::study_planner::{
    StudyCandidate, StudyCost, StudyPlanError, StudyPlanResult, plan_studies,
};

/// Wire schema version.
pub const STUDY_PLAN_ARTIFACT_VERSION: u32 = 1;
/// Artifact kind.
pub const STUDY_PLAN_ARTIFACT_KIND: &str = "study_plan_v1";
/// Rule-set identity of the X1 route recorded in the premises.
const MZ_RULE_SET: &str = "trmz.r443.v1";

impl StudyPlanError {
    fn invalid_artifact(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("transport_not_certified"),
            detail: "study_plan.invalid_artifact",
            message: message.into(),
            receipt: None,
        }
    }

    fn consumer_limits(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("route_not_supported"),
            detail: "study_plan.bounds_exceeded",
            message: message.into(),
            receipt: None,
        }
    }
}

/// One declared candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyCandidateWire {
    /// Candidate id.
    pub id: String,
    /// Population.
    pub population: String,
    /// Interventions.
    pub interventions: Vec<u32>,
    /// Feasible level combinations; `None` is every level.
    pub levels: Option<Vec<Vec<(u32, ValueWire)>>>,
    /// Jointly measured margin.
    pub measured: Vec<u32>,
    /// Recruitment declaration.
    pub recruitment: String,
    /// Cost units.
    pub cost_units: u64,
    /// Sample budget.
    pub sample_budget: u64,
    /// Required companions.
    pub requires: Vec<String>,
    /// Conflicting candidates.
    pub conflicts: Vec<String>,
    /// Feasibility notes.
    pub feasibility_constraints: Vec<String>,
}

/// Plan limits in force.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyPlanLimitsWire {
    /// Operation limit of the one plan budget.
    pub operations: usize,
    /// Depth limit.
    pub depth: usize,
    /// Effective memory cap in bytes.
    pub memory_limit_bytes: u64,
}

/// Everything the plan is a function of, except the data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyPlanPremisesWire {
    /// `mz` or `mixed`.
    pub route: String,
    /// Frozen rule-set version of the route.
    pub rule_set: String,
    /// Ranking version.
    pub ranking: String,
    /// Shared causal graph.
    pub graph: antecedent_io::wire::AdmgWire,
    /// Canonical query (outcomes, treatments, target, declared sources).
    pub query: MixedSourceQueryWire,
    /// Declared candidates, in id order.
    pub candidates: Vec<StudyCandidateWire>,
    /// Plan limits.
    pub limits: StudyPlanLimitsWire,
}

/// Data lineage: the frozen base catalog.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyPlanDataWire {
    /// Canonical base catalog, with every binding's snapshot.
    pub catalog: EvidenceCatalogWire,
}

/// The frozen base failure.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyBaseFailureWire {
    /// Reason code.
    pub code: String,
    /// Route detail.
    pub detail: String,
    /// Stable facts.
    pub facts: Vec<String>,
}

/// One subset's outcome.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudySubsetWire {
    /// Candidate ids.
    pub candidates: Vec<String>,
    /// Total cost units.
    pub cost_units: u64,
    /// Total sample budget.
    pub sample_budget: u64,
    /// Status name.
    pub status: String,
    /// Proposal index (sufficient), code and detail (insufficient), message
    /// (refused), operations (over the decision cap), dominating subset, or
    /// constraint (infeasible).
    pub note: Vec<String>,
}

/// One repaired factor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyRepairWire {
    /// Candidate.
    pub candidate: String,
    /// Regime.
    pub regime: u32,
    /// Population.
    pub population: String,
    /// Interventions.
    pub intervened: Vec<u32>,
    /// Factors `(variables, conditioned_on)` that read the regime.
    pub factors: Vec<(Vec<u32>, Vec<u32>)>,
    /// Proof-derived required margin.
    pub required_margin: Vec<u32>,
    /// Declared margin.
    pub declared_margin: Vec<u32>,
    /// X9 proof steps.
    pub proof_steps: Vec<usize>,
}

/// The checked derivation of a proposal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyDerivationWire {
    /// `mz`, `mixed` or `named`.
    pub kind: String,
    /// X1 derivation record.
    pub mz: Option<MzTransportDerivationRecord>,
    /// X9 derivation record.
    pub mixed: Option<MixedSourceDerivationRecord>,
    /// X9 named route.
    pub named_route: Option<String>,
}

/// One ranked proposal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyProposalWire {
    /// Candidate ids.
    pub candidates: Vec<String>,
    /// Total cost units.
    pub cost_units: u64,
    /// Total sample budget.
    pub sample_budget: u64,
    /// Identifying stage.
    pub stage: String,
    /// Cited regimes.
    pub cited_regimes: Vec<u32>,
    /// Repaired factors.
    pub repairs: Vec<StudyRepairWire>,
    /// Uncited candidates.
    pub uncited_candidates: Vec<String>,
    /// Checked derivation.
    pub derivation: StudyDerivationWire,
    /// Operations of the re-run decision.
    pub decision_operations: usize,
}

/// The receipt of a budget stop.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyPlanStopWire {
    /// `budget` or `proposal_cap`.
    pub kind: String,
    /// `search.<stop>` for a budget stop.
    pub stop: Option<String>,
    /// Operation limit in force.
    pub operations_limit: Option<usize>,
    /// Depth limit in force.
    pub depth_limit: Option<usize>,
    /// Memory cap in force.
    pub memory_limit_bytes: Option<u64>,
    /// Operations consumed.
    pub operations_consumed: Option<usize>,
    /// Depth reached.
    pub depth_reached: Option<usize>,
    /// Explored regions.
    pub explored: Vec<String>,
    /// Unevaluated regions.
    pub unevaluated: Vec<String>,
}

/// The plan outcome.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyPlanOutcomeWire {
    /// Frozen base failure.
    pub failure: StudyBaseFailureWire,
    /// Every subset, in evaluation order.
    pub subsets: Vec<StudySubsetWire>,
    /// Ranked proposals.
    pub proposals: Vec<StudyProposalWire>,
    /// Early end.
    pub stop: Option<StudyPlanStopWire>,
    /// Top proposal certified cost-minimal.
    pub minimal: bool,
    /// Plan status `(code, detail)` when no proposal exists.
    pub status: Option<(String, String)>,
    /// Operations charged by the plan budget.
    pub operations_consumed: usize,
}

/// A durable, independently replayable study plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyPlanArtifactWire {
    /// Schema version.
    pub version: u32,
    /// `study_plan_v1`.
    pub kind: String,
    /// Premises.
    pub premises: StudyPlanPremisesWire,
    /// Data lineage.
    pub data: StudyPlanDataWire,
    /// Outcome.
    pub plan: StudyPlanOutcomeWire,
    /// Digest of the premises.
    pub premises_digest: String,
    /// Digest of the data lineage.
    pub data_digest: String,
    /// Digest of both digests and the outcome.
    pub plan_digest: String,
}

/// Maxima a consumer accepts; stored limits above them refuse before any work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StudyPlanConsumeLimits {
    /// Operation and depth maxima.
    pub search: SearchLimits,
    /// Memory maximum (further lowered by the context's hard limit).
    pub memory_limit_bytes: u64,
}

impl Default for StudyPlanConsumeLimits {
    fn default() -> Self {
        Self { search: STUDY_PLAN_DEFAULT_LIMITS, memory_limit_bytes: STUDY_PLAN_MEMORY_BYTES }
    }
}

fn raw(variables: &[VariableId]) -> Vec<u32> {
    variables.iter().map(|v| v.raw()).collect()
}

fn ids(raw: &[u32]) -> Arc<[VariableId]> {
    raw.iter().copied().map(VariableId::from_raw).collect()
}

fn strings(values: &[Arc<str>]) -> Vec<String> {
    values.iter().map(ToString::to_string).collect()
}

fn arcs(values: &[String]) -> Arc<[Arc<str>]> {
    values.iter().map(|v| Arc::from(v.as_str())).collect()
}

fn candidate_wire(candidate: &StudyCandidate) -> StudyCandidateWire {
    StudyCandidateWire {
        id: candidate.id.to_string(),
        population: candidate.population.to_string(),
        interventions: raw(&candidate.interventions),
        levels: candidate.levels.as_ref().map(|levels| {
            levels
                .iter()
                .map(|level| {
                    level
                        .iter()
                        .map(|a| (a.variable.raw(), ValueWire::from_value(&a.value)))
                        .collect()
                })
                .collect()
        }),
        measured: raw(&candidate.measured),
        recruitment: candidate.recruitment.to_string(),
        cost_units: candidate.cost.units,
        sample_budget: candidate.cost.sample_budget,
        requires: strings(&candidate.requires),
        conflicts: strings(&candidate.conflicts),
        feasibility_constraints: strings(&candidate.feasibility_constraints),
    }
}

fn candidate_from_wire(wire: &StudyCandidateWire) -> StudyCandidate {
    StudyCandidate {
        id: Arc::from(wire.id.as_str()),
        population: Arc::from(wire.population.as_str()),
        interventions: ids(&wire.interventions),
        levels: wire.levels.as_ref().map(|levels| {
            levels
                .iter()
                .map(|level| {
                    level
                        .iter()
                        .map(|(v, x)| InterventionAssignment {
                            variable: VariableId::from_raw(*v),
                            value: x.to_value(),
                        })
                        .collect::<Arc<[_]>>()
                })
                .collect()
        }),
        measured: ids(&wire.measured),
        recruitment: Arc::from(wire.recruitment.as_str()),
        cost: StudyCost { units: wire.cost_units, sample_budget: wire.sample_budget },
        requires: arcs(&wire.requires),
        conflicts: arcs(&wire.conflicts),
        feasibility_constraints: arcs(&wire.feasibility_constraints),
    }
}

fn query_wire(route: &StudyPlanRoute) -> MixedSourceQueryWire {
    match route {
        StudyPlanRoute::Mixed(query) => MixedSourceQueryWire::from_query(query),
        StudyPlanRoute::Mz(query) => MixedSourceQueryWire::from_query(&MixedSourceQuery {
            outcomes: Arc::clone(&query.outcomes),
            treatments: Arc::clone(&query.treatments),
            target: Arc::clone(&query.target),
            sources: Arc::clone(&query.sources),
        }),
    }
}

fn route_from_wire(
    kind: &str,
    query: &MixedSourceQueryWire,
) -> Result<StudyPlanRoute, StudyPlanError> {
    let sources = query
        .sources
        .iter()
        .map(|s: &MzSourceWire| ZTransportSourceSpec {
            population: Arc::from(s.population.as_str()),
            controllable: ids(&s.controllable),
            experiment_assignment: s
                .experiment_assignment
                .iter()
                .map(|(v, x)| InterventionAssignment {
                    variable: VariableId::from_raw(*v),
                    value: x.to_value(),
                })
                .collect(),
            selection_targets: ids(&s.selection_targets),
        })
        .collect::<Arc<[_]>>();
    let (outcomes, treatments, target) =
        (ids(&query.outcomes), ids(&query.treatments), Arc::<str>::from(query.target.as_str()));
    match kind {
        "mz" => Ok(StudyPlanRoute::Mz(MzTransportQuery { outcomes, treatments, target, sources })),
        "mixed" => {
            Ok(StudyPlanRoute::Mixed(MixedSourceQuery { outcomes, treatments, target, sources }))
        }
        other => Err(StudyPlanError::invalid_artifact(format!("unknown route {other:?}"))),
    }
}

const fn rule_set(route: &StudyPlanRoute) -> &'static str {
    match route {
        StudyPlanRoute::Mz(_) => MZ_RULE_SET,
        StudyPlanRoute::Mixed(_) => MIXED_SOURCE_RULE_SET,
    }
}

fn subset_wire(record: &antecedent_identify::StudySubsetRecord) -> StudySubsetWire {
    let note = match &record.outcome {
        StudySubsetOutcome::Sufficient { proposal } => vec![proposal.to_string()],
        StudySubsetOutcome::Insufficient { code, detail } => {
            vec![(*code).to_owned(), (*detail).to_owned()]
        }
        StudySubsetOutcome::Refused { message } => vec![message.clone()],
        StudySubsetOutcome::OverDecisionCap { operations } => vec![operations.to_string()],
        StudySubsetOutcome::Dominated { by } => strings(by),
        StudySubsetOutcome::Infeasible { reason } => vec![(*reason).to_owned()],
        StudySubsetOutcome::Unevaluated => Vec::new(),
    };
    StudySubsetWire {
        candidates: strings(&record.candidates),
        cost_units: record.cost_units,
        sample_budget: record.sample_budget,
        status: record.outcome.status().to_owned(),
        note,
    }
}

fn outcome_wire(plan: &StudyPlan) -> StudyPlanOutcomeWire {
    StudyPlanOutcomeWire {
        failure: StudyBaseFailureWire {
            code: plan.failure.code.to_owned(),
            detail: plan.failure.detail.to_owned(),
            facts: plan.failure.facts.clone(),
        },
        subsets: plan.subsets.iter().map(subset_wire).collect(),
        proposals: plan
            .proposals
            .iter()
            .map(|p| StudyProposalWire {
                candidates: strings(&p.candidates),
                cost_units: p.cost_units,
                sample_budget: p.sample_budget,
                stage: p.stage.clone(),
                cited_regimes: p.cited_regimes.iter().map(|r: &RegimeId| r.raw()).collect(),
                repairs: p
                    .repairs
                    .iter()
                    .map(|r| StudyRepairWire {
                        candidate: r.candidate.to_string(),
                        regime: r.regime.raw(),
                        population: r.population.to_string(),
                        intervened: raw(&r.intervened),
                        factors: r
                            .factors
                            .iter()
                            .map(|f| (raw(&f.variables), raw(&f.conditioned_on)))
                            .collect(),
                        required_margin: raw(&r.required_margin),
                        declared_margin: raw(&r.declared_margin),
                        proof_steps: r.proof_steps.clone(),
                    })
                    .collect(),
                uncited_candidates: strings(&p.uncited_candidates),
                derivation: match &p.derivation {
                    StudyProposalDerivation::Mz(record) => StudyDerivationWire {
                        kind: "mz".into(),
                        mz: Some((**record).clone()),
                        mixed: None,
                        named_route: None,
                    },
                    StudyProposalDerivation::Mixed(record) => StudyDerivationWire {
                        kind: "mixed".into(),
                        mz: None,
                        mixed: Some((**record).clone()),
                        named_route: None,
                    },
                    StudyProposalDerivation::Named { route } => StudyDerivationWire {
                        kind: "named".into(),
                        mz: None,
                        mixed: None,
                        named_route: Some((*route).to_owned()),
                    },
                },
                decision_operations: p.decision_operations,
            })
            .collect(),
        stop: plan.stop.as_ref().map(|stop| match stop {
            StudyPlanStop::Budget(receipt) => StudyPlanStopWire {
                kind: "budget".into(),
                stop: Some(receipt.stop.code().to_owned()),
                operations_limit: Some(receipt.operations_limit),
                depth_limit: Some(receipt.depth_limit),
                memory_limit_bytes: receipt.memory_limit_bytes,
                operations_consumed: receipt.operations_consumed,
                depth_reached: receipt.depth_reached,
                explored: receipt.explored.clone(),
                unevaluated: receipt.unevaluated.clone(),
            },
            StudyPlanStop::ProposalCap => StudyPlanStopWire {
                kind: "proposal_cap".into(),
                stop: None,
                operations_limit: None,
                depth_limit: None,
                memory_limit_bytes: None,
                operations_consumed: None,
                depth_reached: None,
                explored: Vec::new(),
                unevaluated: Vec::new(),
            },
        }),
        minimal: plan.minimal,
        status: plan.status().map(|(code, detail)| (code.to_owned(), detail.to_owned())),
        operations_consumed: plan.operations_consumed,
    }
}

fn digest<T: Serialize>(domain: IdentityDomain, value: &T) -> Result<String, StudyPlanError> {
    antecedent_io::identity::digest_wire(domain, value)
        .map(|d| d.to_hex())
        .map_err(|e| StudyPlanError::invalid_artifact(e.to_string()))
}

impl StudyPlanArtifactWire {
    /// Recompute the premises, data and plan digests of this wire (after an
    /// edit, to re-seal it).
    ///
    /// # Errors
    /// Encoding failure.
    pub fn sealed(mut self) -> Result<Self, StudyPlanError> {
        self.premises_digest = digest(IdentityDomain::Identification, &self.premises)?;
        self.data_digest = digest(IdentityDomain::Identification, &self.data)?;
        self.plan_digest =
            digest(IdentityDomain::Claim, &(&self.premises_digest, &self.data_digest, &self.plan))?;
        Ok(self)
    }

    /// Replay the whole plan under the stored limits and accept only an
    /// identical outcome.
    ///
    /// Stored limits above `limits` (or a stored memory cap above the smaller
    /// of `limits` and the context's hard limit) refuse before any work. The
    /// replay runs under exactly the stored limits, so only cancellation can
    /// make it end differently from the producer's plan: a replay that the
    /// budget stops before the base decision, or that cancellation stops, is
    /// `study_plan.budget` (the artifact is neither accepted nor refuted).
    ///
    /// # Errors
    /// `study_plan.bounds_exceeded` for unaffordable stored limits;
    /// `study_plan.budget` for a replay stopped as above;
    /// `study_plan.invalid_artifact` for another version or kind, a digest that
    /// does not match, undecodable premises or data, a replay the planner
    /// refuses, or a replayed outcome that differs.
    pub fn consume_with_limits(
        &self,
        limits: StudyPlanConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<StudyPlanResult, StudyPlanError> {
        if self.version != STUDY_PLAN_ARTIFACT_VERSION || self.kind != STUDY_PLAN_ARTIFACT_KIND {
            return Err(StudyPlanError::invalid_artifact(
                "unsupported study-plan artifact version",
            ));
        }
        let stored = self.premises.limits;
        let memory_cap = ctx
            .memory
            .hard_limit_bytes
            .map_or(limits.memory_limit_bytes, |h| h.min(limits.memory_limit_bytes));
        if stored.operations > limits.search.operations
            || stored.depth > limits.search.depth
            || stored.memory_limit_bytes > memory_cap
        {
            return Err(StudyPlanError::consumer_limits(
                "the artifact's stored plan limits exceed this consumer's maxima",
            ));
        }
        let resealed = self.clone().sealed()?;
        if resealed.premises_digest != self.premises_digest
            || resealed.data_digest != self.data_digest
            || resealed.plan_digest != self.plan_digest
        {
            return Err(StudyPlanError::invalid_artifact("a study-plan digest does not match"));
        }
        let route = route_from_wire(&self.premises.route, &self.premises.query)?;
        if self.premises.rule_set != rule_set(&route) || self.premises.ranking != STUDY_PLAN_RANKING
        {
            return Err(StudyPlanError::invalid_artifact(
                "the artifact names another rule set or ranking version",
            ));
        }
        let graph = antecedent_io::admg_from_wire(&self.premises.graph)
            .map_err(|e| StudyPlanError::invalid_artifact(e.to_string()))?;
        let catalog = self
            .data
            .catalog
            .to_catalog()
            .map_err(|e| StudyPlanError::invalid_artifact(e.to_string()))?;
        let candidates =
            self.premises.candidates.iter().map(candidate_from_wire).collect::<Vec<_>>();
        let replay_limits = StudyPlanLimits {
            search: SearchLimits { operations: stored.operations, depth: stored.depth },
            memory_limit_bytes: stored.memory_limit_bytes,
        };
        let replayed = match plan_studies(&graph, &route, &catalog, &candidates, replay_limits, ctx)
        {
            Ok(replayed) => replayed,
            // A stop is never a verdict against the artifact.
            Err(error) if error.code == reason_code!("transport_budget_cancel") => {
                return Err(StudyPlanError {
                    message: format!("the replay stopped: {}", error.message),
                    ..error
                });
            }
            Err(error) => {
                return Err(StudyPlanError::invalid_artifact(format!(
                    "the plan does not replay: {error}"
                )));
            }
        };
        if let Some(receipt) = replayed.plan().receipt().filter(|r| r.stop == SearchStop::Cancelled)
        {
            return Err(StudyPlanError::budget(
                "the replay was cancelled before it finished",
                Some(receipt.clone()),
            ));
        }
        let again = replayed.to_artifact()?;
        if again.premises != self.premises || again.data != self.data || again.plan != self.plan {
            return Err(StudyPlanError::invalid_artifact(
                "the replayed plan differs from the stored plan",
            ));
        }
        Ok(replayed)
    }
}

impl StudyPlanResult {
    /// Export the plan as a sealed `study_plan_v1` artifact.
    ///
    /// # Errors
    /// `study_plan.budget` for a plan that cancellation stopped (no consumer
    /// could replay it); graph, catalog or digest encoding failure.
    pub fn to_artifact(&self) -> Result<StudyPlanArtifactWire, StudyPlanError> {
        if self.plan.receipt().is_some_and(|r| r.stop == SearchStop::Cancelled) {
            return Err(StudyPlanError::budget(
                "a plan stopped by cancellation cannot be replayed; plan again to export it",
                None,
            ));
        }
        let catalog = self
            .catalog
            .canonicalized()
            .map_err(|e| StudyPlanError::invalid_artifact(e.to_string()))?;
        let limits = self.plan.limits;
        StudyPlanArtifactWire {
            version: STUDY_PLAN_ARTIFACT_VERSION,
            kind: STUDY_PLAN_ARTIFACT_KIND.to_owned(),
            premises: StudyPlanPremisesWire {
                route: self.plan.route.name().to_owned(),
                rule_set: rule_set(&self.plan.route).to_owned(),
                ranking: STUDY_PLAN_RANKING.to_owned(),
                graph: antecedent_io::admg_to_wire(&self.graph)
                    .map_err(|e| StudyPlanError::invalid_artifact(e.to_string()))?,
                query: query_wire(&self.plan.route),
                candidates: self.candidates.iter().map(candidate_wire).collect(),
                limits: StudyPlanLimitsWire {
                    operations: limits.search.operations,
                    depth: limits.search.depth,
                    memory_limit_bytes: limits.memory_limit_bytes,
                },
            },
            data: StudyPlanDataWire { catalog: EvidenceCatalogWire::from_catalog(&catalog) },
            plan: outcome_wire(&self.plan),
            premises_digest: String::new(),
            data_digest: String::new(),
            plan_digest: String::new(),
        }
        .sealed()
    }
}

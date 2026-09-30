//! Declared candidate studies for X6 study planning over the X1 (mz) and X9
//! (mixed-source) catalogs.
//!
//! A [`StudyCandidate`] is what a user can run: a population, an intervention
//! set with its feasible levels (or every level), the jointly measured margin,
//! a recruitment declaration, an integer cost and its subset constraints.
//! [`plan_studies`] compiles each one to a typed hypothetical
//! [`EvidenceCatalogDelta`] (deterministic regime ids, in candidate-id order,
//! after the base catalog's largest id) and runs the bounded planner of
//! `antecedent_identify::plan_study_additions`, which owns the one shared
//! search budget. Nothing here claims a probability of success: a sufficient
//! proposal says that if the study delivers exactly the declared regimes (with
//! positive probability at every level the formula reads), the same route
//! would identify the query. [`StudyPlanProposal::receive`] checks the
//! arriving catalog against the proposal and re-identifies it through the
//! public X1/X9 route; a planning preview is never turned into a result.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceCatalogDelta, EvidenceKind, EvidenceRegime,
    ExecutionContext, InterventionAssignment, NodeRef, RegimeId, RegimeKind, VariableId,
    reason_code,
};
use antecedent_graph::Admg;
use antecedent_identify::{
    MIXED_SOURCE_DEFAULT_LIMITS, MZ_TRANSPORT_DEFAULT_LIMITS, MixedSourceDecision,
    MzTransportDecision, StudyPlan, StudyPlanCandidate, StudyPlanLimits, StudyPlanRefusal,
    StudyPlanRoute, StudyProposal, decide_mixed_source, decide_mz_transport, plan_study_additions,
};

use crate::plan_common::{
    valid_intervention_values, validate_actual_delta, validate_base_preserved,
};

/// Declared cost of a candidate study, in caller-defined integer units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StudyCost {
    /// Cost units, at least one; ranking sums them exactly.
    pub units: u64,
    /// Sample budget; a ranking tie-breaker, never a sufficiency input.
    pub sample_budget: u64,
}

/// One declared candidate study.
#[derive(Clone, Debug, PartialEq)]
pub struct StudyCandidate {
    /// Stable identity (1-64 characters of `[A-Za-z0-9_.:-]`).
    pub id: Arc<str>,
    /// Population the study is collected in: the target or a declared source.
    pub population: Arc<str>,
    /// Hard-intervention set; empty for an observational study.
    pub interventions: Arc<[VariableId]>,
    /// Feasible level combinations, each assigning every intervention once
    /// (one regime per combination). `None` runs every level (one unrestricted
    /// family regime).
    pub levels: Option<Arc<[Arc<[InterventionAssignment]>]>>,
    /// Jointly measured margin (without the interventions).
    pub measured: Arc<[VariableId]>,
    /// Recruitment and sampling declaration; recorded, never a sufficiency input.
    pub recruitment: Arc<str>,
    /// Declared cost.
    pub cost: StudyCost,
    /// Candidates that must run together with this one.
    pub requires: Arc<[Arc<str>]>,
    /// Candidates that cannot run together with this one.
    pub conflicts: Arc<[Arc<str>]>,
    /// Feasibility notes, recorded with the proposal.
    pub feasibility_constraints: Arc<[Arc<str>]>,
}

/// A refused plan, proposal or arrival: a registered reason code and a stable
/// `study_plan.*` detail.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct StudyPlanError {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `study_plan.*` detail.
    pub detail: &'static str,
    /// Human-readable explanation.
    pub message: String,
    /// The plan budget's receipt, for a budget stop.
    pub receipt: Option<Box<antecedent_core::SearchReceipt>>,
}

impl From<StudyPlanRefusal> for StudyPlanError {
    fn from(refusal: StudyPlanRefusal) -> Self {
        Self {
            code: refusal.code,
            detail: refusal.detail,
            message: refusal.message,
            receipt: refusal.receipt,
        }
    }
}

impl StudyPlanError {
    pub(crate) fn invalid_candidate(message: impl Into<String>) -> Self {
        StudyPlanRefusal::invalid_candidate(message).into()
    }

    /// An arriving catalog that does not match its proposal.
    pub(crate) fn arrival_mismatch(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("invalid_argument"),
            detail: "study_plan.arrival_mismatch",
            message: message.into(),
            receipt: None,
        }
    }

    fn arrival_not_identified(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("transport_not_certified"),
            detail: "study_plan.arrival_not_identified",
            message: message.into(),
            receipt: None,
        }
    }
}

/// A finished plan with the declarations and catalog it was made from.
#[derive(Clone, Debug)]
pub struct StudyPlanResult {
    pub(crate) graph: Admg,
    pub(crate) catalog: EvidenceCatalog,
    pub(crate) candidates: Vec<StudyCandidate>,
    pub(crate) compiled: Vec<StudyPlanCandidate>,
    pub(crate) plan: StudyPlan,
}

impl StudyPlanResult {
    /// The plan: frozen failure, every subset, ranked proposals, stop receipt
    /// and minimality flag.
    #[must_use]
    pub const fn plan(&self) -> &StudyPlan {
        &self.plan
    }

    /// Declared candidates, in id order.
    #[must_use]
    pub fn candidates(&self) -> &[StudyCandidate] {
        &self.candidates
    }

    /// Each candidate compiled to its hypothetical delta, in id order.
    #[must_use]
    pub fn compiled(&self) -> &[StudyPlanCandidate] {
        &self.compiled
    }

    /// The base catalog the plan was frozen against.
    #[must_use]
    pub const fn catalog(&self) -> &EvidenceCatalog {
        &self.catalog
    }

    /// The proposal at `rank` (0 is the cheapest), ready to receive evidence.
    #[must_use]
    pub fn proposal(&self, rank: usize) -> Option<StudyPlanProposal> {
        let proposal = self.plan.proposals.get(rank)?.clone();
        let regimes = self
            .compiled
            .iter()
            .filter(|c| proposal.candidates.contains(&c.id))
            .flat_map(|c| c.delta.proposed_regimes.iter().cloned())
            .collect::<Vec<_>>();
        Some(StudyPlanProposal {
            graph: self.graph.clone(),
            route: self.plan.route.clone(),
            catalog: self.catalog.clone(),
            delta: EvidenceCatalogDelta { proposed_regimes: regimes.into() },
            proposal,
        })
    }
}

/// One sufficient proposal bound to its frozen base catalog.
#[derive(Clone, Debug)]
pub struct StudyPlanProposal {
    graph: Admg,
    route: StudyPlanRoute,
    catalog: EvidenceCatalog,
    delta: EvidenceCatalogDelta,
    proposal: StudyProposal,
}

/// The public route's decision on the arriving catalog.
#[derive(Clone, Debug)]
pub enum StudyArrivalDecision {
    /// An X1 `Identified` decision.
    Mz(Box<MzTransportDecision>),
    /// An X9 `Identified` decision, or the named route that identifies.
    Mixed(Box<MixedSourceDecision>),
}

/// Evidence that arrived for a proposal and was re-identified.
#[derive(Clone, Debug)]
pub struct StudyArrival {
    /// The public route's decision on the real, available catalog.
    pub decision: StudyArrivalDecision,
    /// The provider snapshot that binds every arriving regime.
    pub provider_snapshot: Arc<str>,
}

impl StudyPlanProposal {
    /// The planned proposal.
    #[must_use]
    pub const fn proposal(&self) -> &StudyProposal {
        &self.proposal
    }

    /// The regimes the proposal needs to arrive, as a hypothetical delta.
    #[must_use]
    pub const fn delta(&self) -> &EvidenceCatalogDelta {
        &self.delta
    }

    /// Accept arriving evidence: the actual catalog must keep the frozen base
    /// unchanged and hold every proposed regime as available evidence with
    /// exactly the proposed shape, bound to `provider_snapshot`; then the
    /// public route (`decide_mz_transport` or `decide_mixed_source` under its
    /// default limits) decides the real catalog from scratch. Support
    /// (positivity) is not checked here: the ordinary evaluator refuses a law
    /// without mass at a level the formula reads.
    ///
    /// # Errors
    /// `study_plan.arrival_mismatch` for a catalog that does not match the
    /// proposal; `study_plan.arrival_not_identified` when the public route does
    /// not identify the arriving catalog.
    pub fn receive(
        &self,
        actual: &EvidenceCatalog,
        provider_snapshot: impl Into<Arc<str>>,
        ctx: &ExecutionContext,
    ) -> Result<StudyArrival, StudyPlanError> {
        actual.validate().map_err(|e| StudyPlanError::arrival_mismatch(e.to_string()))?;
        // Compare canonical forms: declaration order is not identity.
        let canonical = |catalog: &EvidenceCatalog| {
            catalog.canonicalized().map_err(|e| StudyPlanError::arrival_mismatch(e.to_string()))
        };
        validate_base_preserved(&canonical(&self.catalog)?, &canonical(actual)?)
            .map_err(StudyPlanError::arrival_mismatch)?;
        validate_actual_delta(&self.delta, actual).map_err(StudyPlanError::arrival_mismatch)?;
        let provider_snapshot = provider_snapshot.into();
        if provider_snapshot.trim().is_empty()
            || !self.delta.proposed_regimes.iter().all(|regime| {
                actual.bindings.iter().any(|binding| {
                    binding.regime == regime.id
                        && binding.snapshot_identity.as_ref() == provider_snapshot.as_ref()
                })
            })
        {
            return Err(StudyPlanError::arrival_mismatch(
                "the provider snapshot does not bind every arriving proposed regime",
            ));
        }
        let decision = match &self.route {
            StudyPlanRoute::Mz(query) => {
                match decide_mz_transport(
                    &self.graph,
                    query,
                    actual,
                    MZ_TRANSPORT_DEFAULT_LIMITS,
                    ctx,
                ) {
                    Ok(decision @ MzTransportDecision::Identified { .. }) => {
                        StudyArrivalDecision::Mz(Box::new(decision))
                    }
                    Ok(other) => {
                        return Err(StudyPlanError::arrival_not_identified(format!(
                            "the mz route does not identify the arriving catalog ({})",
                            other.detail_code().unwrap_or_default()
                        )));
                    }
                    Err(error) => {
                        return Err(StudyPlanError::arrival_not_identified(error.to_string()));
                    }
                }
            }
            StudyPlanRoute::Mixed(query) => {
                match decide_mixed_source(
                    &self.graph,
                    query,
                    actual,
                    MIXED_SOURCE_DEFAULT_LIMITS,
                    ctx,
                ) {
                    Ok(
                        decision @ (MixedSourceDecision::Identified { .. }
                        | MixedSourceDecision::NamedRoute { .. }),
                    ) => StudyArrivalDecision::Mixed(Box::new(decision)),
                    Ok(other) => {
                        return Err(StudyPlanError::arrival_not_identified(format!(
                            "the mixed route does not identify the arriving catalog ({})",
                            other.detail_code().unwrap_or_default()
                        )));
                    }
                    Err(error) => {
                        return Err(StudyPlanError::arrival_not_identified(error.to_string()));
                    }
                }
            }
        };
        Ok(StudyArrival { decision, provider_snapshot })
    }
}

fn graph_variables(graph: &Admg) -> BTreeSet<VariableId> {
    graph
        .nodes()
        .iter()
        .filter_map(|node| match node {
            NodeRef::Static(variable) => Some(*variable),
            _ => None,
        })
        .collect()
}

/// Compile declared candidates (sorted by id) to neutral planning candidates:
/// one proposed regime per declared level combination (or one unrestricted
/// family), with ids allocated after the base catalog's largest regime id.
fn compile(
    graph: &Admg,
    catalog: &EvidenceCatalog,
    candidates: &[StudyCandidate],
) -> Result<Vec<StudyPlanCandidate>, StudyPlanError> {
    let variables = graph_variables(graph);
    let mut next = catalog.regimes.iter().map(|r| r.id.raw()).max().map_or(0, |m| m + 1);
    let mut out = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let who = &candidate.id;
        let distinct = |vs: &[VariableId]| vs.iter().collect::<BTreeSet<_>>().len() == vs.len();
        if candidate.recruitment.trim().is_empty()
            || candidate.measured.is_empty()
            || !distinct(&candidate.measured)
            || !distinct(&candidate.interventions)
            || candidate.measured.iter().any(|v| candidate.interventions.contains(v))
            || !candidate
                .measured
                .iter()
                .chain(candidate.interventions.iter())
                .all(|v| variables.contains(v))
        {
            return Err(StudyPlanError::invalid_candidate(format!(
                "candidate {who} needs a recruitment declaration and a non-empty margin of \
                 distinct graph variables disjoint from its interventions"
            )));
        }
        let kind = if candidate.interventions.is_empty() {
            RegimeKind::Observational
        } else {
            RegimeKind::Experimental
        };
        let levels: Vec<Arc<[InterventionAssignment]>> = match &candidate.levels {
            None => vec![Arc::from([])],
            Some(levels) if kind == RegimeKind::Experimental && !levels.is_empty() => {
                levels.to_vec()
            }
            Some(_) => {
                return Err(StudyPlanError::invalid_candidate(format!(
                    "candidate {who} declares levels without interventions, or an empty level list"
                )));
            }
        };
        let mut regimes = Vec::with_capacity(levels.len());
        for (k, level) in levels.iter().enumerate() {
            let id = RegimeId::from_raw(next);
            next = next.checked_add(1).ok_or_else(|| {
                StudyPlanError::invalid_candidate("regime ids overflow".to_owned())
            })?;
            let mut regime = EvidenceRegime::try_new(
                id,
                kind,
                EvidenceKind::Proposed,
                Arc::clone(&candidate.interventions),
                Arc::clone(level),
                Arc::clone(&candidate.measured),
                Arc::clone(&candidate.population),
                DistributionAvailability::Joint,
            )
            .map_err(|e| StudyPlanError::invalid_candidate(format!("candidate {who}: {e}")))?;
            regime.label = Some(Arc::from(format!("{who}#{k}")));
            regime.study = Some(Arc::clone(who));
            if !level.is_empty() && !valid_intervention_values(catalog, &regime) {
                return Err(StudyPlanError::invalid_candidate(format!(
                    "candidate {who} level {k} must assign every intervention once, inside the \
                     domain its population declares"
                )));
            }
            regimes.push(regime);
        }
        let delta = EvidenceCatalogDelta::try_new(catalog, regimes)
            .map_err(|e| StudyPlanError::invalid_candidate(format!("candidate {who}: {e}")))?;
        out.push(StudyPlanCandidate {
            id: Arc::clone(who),
            delta,
            cost_units: candidate.cost.units,
            sample_budget: candidate.cost.sample_budget,
            requires: Arc::clone(&candidate.requires),
            conflicts: Arc::clone(&candidate.conflicts),
        });
    }
    Ok(out)
}

/// Compile the declared candidates and plan the cheapest subset of at most
/// three that would make the failed X1 or X9 decision identify, under one
/// shared search budget (see `antecedent_identify::plan_study_additions`).
/// The result is invariant to candidate and source declaration order.
///
/// # Errors
/// Every [`StudyPlanRefusal`] of the planner, plus
/// `study_plan.invalid_candidate` for a declaration that does not compile
/// (empty or overlapping margin, a variable outside the graph, levels that do
/// not assign every intervention or leave the declared domain, no recruitment
/// declaration).
pub fn plan_studies(
    graph: &Admg,
    route: &StudyPlanRoute,
    catalog: &EvidenceCatalog,
    candidates: &[StudyCandidate],
    limits: StudyPlanLimits,
    ctx: &ExecutionContext,
) -> Result<StudyPlanResult, StudyPlanError> {
    let mut sorted = candidates.to_vec();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));
    let compiled = compile(graph, catalog, &sorted)?;
    let plan = plan_study_additions(graph, route, catalog, &compiled, limits, ctx)?;
    Ok(StudyPlanResult {
        graph: graph.clone(),
        catalog: catalog.clone(),
        candidates: sorted,
        compiled,
        plan,
    })
}

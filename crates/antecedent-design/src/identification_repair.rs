//! Identification repair families for the identifiers beyond back-door adjustment.
//!
//! A failed identification owes evidence that depends on which identifier could
//! succeed with it. Each family here freezes one such contract and lets the
//! generic [`crate::repair()`] search ask the identifier's own theorem whether a
//! proposed study would repair it:
//!
//! * [`FrontDoorRepairFamily`] needs one joint observational law over the
//!   treatment, the outcome and a mediator set that intercepts every directed
//!   path and is shielded from the treatment's confounding. The mediator set is
//!   re-verified by d-separation, independently of the identifier.
//! * [`InstrumentRepairFamily`] needs one joint observational law over an
//!   instrument, the treatment and the outcome. Relevance and exclusion are
//!   `EstablishAssumption` obligations: a study supplies the data the Wald ratio
//!   is read from, it never proves the instrument valid, so a study alone never
//!   certifies this family.
//! * [`HedgeRepairFamily`] answers a hedge (a proof that observation cannot
//!   identify the effect) with experiments: setting the treatment measures the
//!   interventional law directly, and setting a hedge node that intercepts every
//!   directed path from treatment to outcome identifies it through
//!   `P(y | do(x)) = sum_w P(w | do(x)) P(y | do(w))` when `P(w | do(x))` is
//!   itself identified from observation.
//! * [`SensitivityRepairFamily`] answers a sensitivity analysis whose conclusion
//!   an unmeasured confounder of the reported strength would overturn: measure
//!   that confounder jointly with the treatment and the outcome so the back-door
//!   criterion can adjust for it.
//!
//! Every family refuses `identification_repair.not_a_failure` when the law that
//! is already observed identifies the effect, keeps unresolved obligations that
//! retain their proof step, and reports a budget stop as a resource outcome, not
//! as impossibility.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, sync::Arc};

use antecedent_contract::{
    EvidenceObligation, EvidenceObligationKind, EvidenceObligationSpec, ObligationProvenance,
    ObligationRegime, ObligationScope,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, DistributionAvailability, EvidenceRegime, ExecutionContext,
    IdentificationStatus, NodeRef, VariableId, is_blank,
};
use antecedent_graph::{Admg, BitSet, DSeparationWorkspace, Dag, DenseNodeId, GraphWorkspace};
use antecedent_identify::{
    FrontDoorIdentifier, FrontDoorSearchConfig, HedgeCertificate, IdIdentifier,
    IdentificationWorkspace, InstrumentSearchConfig, InstrumentalVariableIdentifier,
};

use crate::plan_common::id_list;
use crate::repair::{
    BackdoorRepairFamily, FailedContract, FamilyVerdict, RepairCancelled, RepairDerivation,
    RepairError, RepairFamily, dense_of, graph_variables,
};
use crate::study_candidate::DurableStudyCandidate;

mod hedge;
mod sensitivity;
pub use hedge::HedgeRepairFamily;
pub use sensitivity::{SensitivityFinding, SensitivityMethod, SensitivityRepairFamily};

/// Most candidate mediator sets or instruments one search collects.
const MAX_CANDIDATES: usize = 4096;
/// Mediator subsets the front-door search may enumerate: the work budget that sets how deep it
/// goes.
const MEDIATOR_SEARCH_SUBSETS: u128 = 1 << 20;

/// Largest mediator set the front-door search enumerates over `pool` candidate nodes: every set
/// size while the number of subsets stays within [`MEDIATOR_SEARCH_SUBSETS`] (so a graph of up
/// to twenty nodes is searched in full), and below that the deepest size the budget affords.
fn max_mediator_set_size(pool: usize) -> usize {
    let mut subsets: u128 = 1;
    let mut choose: u128 = 1;
    let mut size = 0;
    while size < pool {
        choose = choose * (pool - size) as u128 / (size + 1) as u128;
        if subsets.saturating_add(choose) > MEDIATOR_SEARCH_SUBSETS {
            break;
        }
        subsets += choose;
        size += 1;
    }
    size.max(1)
}

fn invalid(message: impl Into<String>) -> RepairError {
    RepairError::invalid_request(message)
}

// ------------------------------------------------------------ graph helpers

/// Nodes reachable from `from` by directed edges that never enter `avoid`.
fn reaches_avoiding(
    node_count: usize,
    children: &dyn Fn(DenseNodeId) -> Vec<DenseNodeId>,
    from: DenseNodeId,
    to: DenseNodeId,
    avoid: &[DenseNodeId],
) -> bool {
    let mut seen = vec![false; node_count];
    let mut stack = vec![from];
    while let Some(node) = stack.pop() {
        for child in children(node) {
            if avoid.contains(&child) || seen[child.as_usize()] {
                continue;
            }
            if child == to {
                return true;
            }
            seen[child.as_usize()] = true;
            stack.push(child);
        }
    }
    false
}

/// A copy of `dag` without the outgoing edges of `nodes`.
fn cut_outgoing(dag: &Dag, nodes: &[DenseNodeId]) -> Option<Dag> {
    let mut out = Dag::empty();
    for node in dag.nodes() {
        out.add_node(*node).ok()?;
    }
    for (from, to) in dag.directed_edge_pairs_sorted() {
        let from = DenseNodeId::from_raw(from);
        if !nodes.contains(&from) {
            out.insert_directed(from, DenseNodeId::from_raw(to)).ok()?;
        }
    }
    Some(out)
}

fn dag_children(dag: &Dag) -> impl Fn(DenseNodeId) -> Vec<DenseNodeId> + '_ {
    move |node| dag.children(node).to_vec()
}

/// The front-door criterion, re-checked without the identifier: `mediators`
/// intercept every directed path from `treatment` to `outcome`, the treatment is
/// d-separated from each mediator once its outgoing edges are cut, and each
/// mediator is d-separated from the outcome given the treatment once the
/// mediators' outgoing edges are cut.
fn verify_frontdoor(
    dag: &Dag,
    treatment: VariableId,
    outcome: VariableId,
    mediators: &[VariableId],
) -> bool {
    let (Some(t), Some(y)) = (dense_of(dag, treatment), dense_of(dag, outcome)) else {
        return false;
    };
    let Some(m) = mediators.iter().map(|v| dense_of(dag, *v)).collect::<Option<Vec<_>>>() else {
        return false;
    };
    if m.is_empty() || m.contains(&t) || m.contains(&y) {
        return false;
    }
    if reaches_avoiding(dag.node_count(), &dag_children(dag), t, y, &m) {
        return false;
    }
    let mut workspace = DSeparationWorkspace::default();
    let Some(from_t) = cut_outgoing(dag, &[t]) else { return false };
    if !m.iter().all(|&node| from_t.is_d_separated(t, node, &[], &mut workspace).unwrap_or(false)) {
        return false;
    }
    let Some(from_m) = cut_outgoing(dag, &m) else { return false };
    m.iter().all(|&node| from_m.is_d_separated(node, y, &[t], &mut workspace).unwrap_or(false))
}

/// The instrument conditions, re-checked without the identifier: `instrument`
/// is not a descendant of the treatment, is associated with it, and is
/// d-separated from the outcome once the treatment's outgoing edges are cut.
fn verify_instrument(
    dag: &Dag,
    treatment: VariableId,
    outcome: VariableId,
    instrument: VariableId,
) -> bool {
    let (Some(t), Some(y), Some(z)) =
        (dense_of(dag, treatment), dense_of(dag, outcome), dense_of(dag, instrument))
    else {
        return false;
    };
    if z == t || z == y {
        return false;
    }
    let mut descendants = BitSet::with_len(dag.node_count());
    dag.descendants_of(&[t], &mut descendants, &mut GraphWorkspace::default());
    if descendants.contains(z) {
        return false;
    }
    let mut workspace = DSeparationWorkspace::default();
    if dag.is_d_separated(z, t, &[], &mut workspace).unwrap_or(true) {
        return false;
    }
    cut_outgoing(dag, &[t])
        .is_some_and(|cut| cut.is_d_separated(z, y, &[], &mut workspace).unwrap_or(false))
}

/// A common cause the graph leaves unmeasured: one per bidirected edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LatentCause {
    /// The variable id standing for the unmeasured common cause (past every graph variable).
    pub variable: VariableId,
    /// The two graph variables it causes.
    pub between: (VariableId, VariableId),
}

/// The directed acyclic graph that latent-projects to `admg`: one new node per
/// bidirected edge, a parent of both endpoints. Identification in the ADMG and
/// in this expansion coincide, and the new nodes name what would have to be
/// measured to adjust for the bidirected confounding.
///
/// # Errors
/// `identification_repair.invalid_request` for a graph with a non-static node.
pub fn latent_expansion(admg: &Admg) -> Result<(Dag, Vec<LatentCause>), RepairError> {
    let mut variables = Vec::with_capacity(admg.node_count());
    for node in admg.nodes() {
        match node {
            NodeRef::Static(variable) => variables.push(*variable),
            _ => return Err(invalid("a failure graph holds only static variables")),
        }
    }
    let mut dag = admg.directed_part().map_err(|e| invalid(e.to_string()))?;
    let mut next = variables.iter().map(|v| v.raw()).max().map_or(0, |m| m.saturating_add(1));
    let mut latents = Vec::new();
    for (a, b) in admg.bidirected_edge_pairs_sorted() {
        let cause = VariableId::from_raw(next);
        next = next.saturating_add(1);
        let node = dag.add_node(NodeRef::Static(cause)).map_err(|e| invalid(e.to_string()))?;
        for end in [a, b] {
            dag.insert_directed(node, DenseNodeId::from_raw(end))
                .map_err(|e| invalid(e.to_string()))?;
        }
        let static_of = |raw: u32| {
            admg.static_variable(DenseNodeId::from_raw(raw)).ok_or_else(|| invalid("unknown node"))
        };
        latents.push(LatentCause { variable: cause, between: (static_of(a)?, static_of(b)?) });
    }
    Ok((dag, latents))
}

// ---------------------------------------------------------------- failures

/// What every graph-route family freezes: the graph (latent causes included),
/// the treatment and outcome, and the law that is observed today.
#[derive(Clone, Debug)]
struct Failure {
    dag: Dag,
    treatment: VariableId,
    outcome: VariableId,
    population: Arc<str>,
    base_law: BTreeSet<VariableId>,
}

impl Failure {
    fn try_new(
        dag: Dag,
        treatment: VariableId,
        outcome: VariableId,
        population: impl Into<Arc<str>>,
        base_law: impl IntoIterator<Item = VariableId>,
    ) -> Result<Self, RepairError> {
        let population = population.into();
        let variables = graph_variables(&dag);
        let base_law: BTreeSet<VariableId> = base_law.into_iter().collect();
        if treatment == outcome
            || !variables.contains(&treatment)
            || !variables.contains(&outcome)
            || base_law.iter().any(|v| !variables.contains(v))
            || is_blank(&population)
        {
            return Err(invalid(
                "an identification contract names a population and a treatment and outcome that \
                 differ and lie in the graph",
            ));
        }
        Ok(Self { dag, treatment, outcome, population, base_law })
    }

    fn usable(&self, law: &BTreeSet<VariableId>) -> bool {
        law.contains(&self.treatment) && law.contains(&self.outcome)
    }

    fn contract(&self, route: &str) -> String {
        format!(
            "{route}:{}:treatment={}:outcome={}",
            self.population,
            self.treatment.raw(),
            self.outcome.raw()
        )
    }

    /// The observational, unconditional joint laws a set of studies would
    /// deliver in the contract's population, with the base law first.
    fn laws_of_studies(
        &self,
        studies: &[&DurableStudyCandidate],
    ) -> Result<Vec<(String, BTreeSet<VariableId>)>, Vec<String>> {
        let mut laws = Vec::new();
        if self.usable(&self.base_law) {
            laws.push(("base".to_owned(), self.base_law.clone()));
        }
        let mut malformed = Vec::new();
        for study in studies {
            if let Err(error) = study.validate() {
                malformed.push(error.to_string());
                continue;
            }
            for (k, evidence) in study.expected_evidence.iter().enumerate() {
                let law: BTreeSet<VariableId> = evidence.measured.iter().copied().collect();
                if evidence.population == self.population
                    && evidence.interventions.is_empty()
                    && evidence.conditioned_on.is_empty()
                    && matches!(evidence.distribution, DistributionAvailability::Joint)
                    && self.usable(&law)
                {
                    laws.push((format!("{}#{k}", study.semantic_id()), law));
                }
            }
        }
        if malformed.is_empty() { Ok(laws) } else { Err(malformed) }
    }

    /// The same laws read from already-proposed regimes.
    fn laws_of_regimes(&self, regimes: &[EvidenceRegime]) -> Vec<(String, BTreeSet<VariableId>)> {
        let mut laws = Vec::new();
        if self.usable(&self.base_law) {
            laws.push(("base".to_owned(), self.base_law.clone()));
        }
        for regime in regimes {
            let law: BTreeSet<VariableId> = regime.measured.iter().copied().collect();
            if regime.population == self.population
                && regime.interventions.is_empty()
                && regime.conditioned_on.is_empty()
                && matches!(regime.distribution, DistributionAvailability::Joint)
                && self.usable(&law)
            {
                laws.push((format!("regime:{}", regime.id.raw()), law));
            }
        }
        laws
    }
}

/// The proposed regimes `studies` would deliver, with consecutive ids from zero
/// (these contracts have no base catalog).
fn regimes_of(studies: &[&DurableStudyCandidate]) -> Result<Vec<EvidenceRegime>, Vec<String>> {
    let mut next = 0_u32;
    let mut regimes = Vec::new();
    for study in studies {
        let produced = study.proposed_regimes(next).map_err(|e| vec![e.to_string()])?;
        next = next.saturating_add(u32::try_from(produced.len()).unwrap_or(u32::MAX));
        regimes.extend(produced);
    }
    Ok(regimes)
}

// -------------------------------------------------------------- obligations

struct Build<'a> {
    family: &'a str,
    source: &'a str,
}

impl Build<'_> {
    fn provenance(&self, step: String) -> ObligationProvenance {
        ObligationProvenance {
            family: Arc::from(self.family),
            source: Arc::from(self.source),
            proof_step: Some(Arc::from(step)),
        }
    }

    fn joint_law(
        &self,
        variables: &BTreeSet<VariableId>,
        population: &Arc<str>,
        reason: String,
        slot: &str,
        step: String,
    ) -> Result<EvidenceObligation, RepairError> {
        EvidenceObligation::try_new(EvidenceObligationSpec {
            quantities: std::collections::BTreeMap::new(),
            kind: EvidenceObligationKind::ProvideJointLaw,
            scope: ObligationScope::Factor,
            variables: variables.iter().copied().collect::<Vec<_>>().into(),
            population: Some(Arc::clone(population)),
            regime: ObligationRegime::observational(true),
            reason: Arc::from(reason),
            required_slots: Arc::from([Arc::from(slot)]),
            min_additional_samples: None,
            provenance: self.provenance(step),
        })
        .map_err(|e| invalid(e.to_string()))
    }

    fn assumption(
        &self,
        reason: String,
        slot: String,
        step: String,
    ) -> Result<EvidenceObligation, RepairError> {
        EvidenceObligation::try_new(EvidenceObligationSpec {
            quantities: std::collections::BTreeMap::new(),
            kind: EvidenceObligationKind::EstablishAssumption,
            scope: ObligationScope::Program,
            variables: Arc::from([]),
            population: None,
            regime: ObligationRegime::observational(false),
            reason: Arc::from(reason),
            required_slots: Arc::from([Arc::from(slot)]),
            min_additional_samples: None,
            provenance: self.provenance(step),
        })
        .map_err(|e| invalid(e.to_string()))
    }

    fn experiment(
        &self,
        target: VariableId,
        outcome: VariableId,
        population: &Arc<str>,
        reason: String,
        slot: String,
        step: String,
    ) -> Result<EvidenceObligation, RepairError> {
        EvidenceObligation::try_new(EvidenceObligationSpec {
            quantities: std::collections::BTreeMap::new(),
            kind: EvidenceObligationKind::Intervene,
            scope: ObligationScope::Factor,
            variables: Arc::from([outcome]),
            population: Some(Arc::clone(population)),
            regime: ObligationRegime {
                interventions: Arc::from([target]),
                conditioned_on: Arc::from([]),
                joint: false,
            },
            reason: Arc::from(reason),
            required_slots: Arc::from([Arc::from(slot)]),
            min_additional_samples: None,
            provenance: self.provenance(step),
        })
        .map_err(|e| invalid(e.to_string()))
    }
}

/// The panel alternative to measuring unmeasured causes: repeated measurements of
/// the same units, whose within-unit comparisons remove confounding by causes
/// that do not change over the observation window. The time-invariance of those
/// causes is an assumption no study establishes.
pub(crate) fn panel_obligations(
    treatment: VariableId,
    outcome: VariableId,
    population: &Arc<str>,
    unmeasured: &[VariableId],
) -> Result<Vec<EvidenceObligation>, RepairError> {
    let source =
        format!("panel:{population}:treatment={}:outcome={}", treatment.raw(), outcome.raw());
    let build = Build { family: "panel", source: &source };
    let list = id_list(unmeasured);
    let mut variables = BTreeSet::new();
    variables.extend([treatment, outcome]);
    Ok(vec![
        build.joint_law(
            &variables,
            population,
            format!(
                "repeated measurements of the same units (a panel) of the treatment and the                  outcome allow within-unit comparisons that remove confounding by the unmeasured                  causes {list} when they do not change over the observation window"
            ),
            "panel:repeated_measures",
            format!("backdoor.unmeasured:{list}"),
        )?,
        build.assumption(
            format!(
                "the unmeasured causes {list} are constant over the observation window; a                  panel cannot establish this"
            ),
            "panel:time_invariant_confounding".to_owned(),
            format!("panel.time_invariance:{list}"),
        )?,
    ])
}

// --------------------------------------------- front-door and instrument core

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Route {
    FrontDoor,
    Instrument,
}

/// What a route's identifier made of one joint law.
enum Analysis {
    Identified(Vec<VariableId>),
    NotIdentified(String),
    Failed(String),
}

impl Route {
    const fn name(self) -> &'static str {
        match self {
            Self::FrontDoor => "frontdoor",
            Self::Instrument => "instrument",
        }
    }

    const fn checker(self) -> &'static str {
        match self {
            Self::FrontDoor => "frontdoor.adjustment",
            Self::Instrument => "instrument.wald",
        }
    }

    const fn witness(self) -> &'static str {
        match self {
            Self::FrontDoor => "mediator_set",
            Self::Instrument => "instrument",
        }
    }

    /// Every set (a mediator set, or one instrument) the identifier finds in the
    /// whole graph and the independent check confirms, and whether the search
    /// was bounded.
    fn candidates(self, failure: &Failure) -> Result<(Vec<Vec<VariableId>>, bool), String> {
        let query = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            failure.treatment,
            failure.outcome,
        ));
        let mut workspace = IdentificationWorkspace::default();
        match self {
            Self::FrontDoor => {
                let identifier = FrontDoorIdentifier {
                    config: FrontDoorSearchConfig {
                        max_results: MAX_CANDIDATES,
                        max_mediator_set_size: max_mediator_set_size(failure.dag.node_count()),
                    },
                };
                let prepared = identifier.prepare(&failure.dag).map_err(|e| e.to_string())?;
                let result = identifier
                    .identify(&prepared, &query, &mut workspace)
                    .map_err(|e| e.to_string())?;
                let bounded =
                    result.diagnostics.iter().any(|d| d.code.ends_with(".search_bounded"));
                let sets = result
                    .estimands
                    .iter()
                    .map(|e| e.mediators.to_vec())
                    .filter(|set| {
                        verify_frontdoor(&failure.dag, failure.treatment, failure.outcome, set)
                    })
                    .collect();
                Ok((sets, bounded))
            }
            Self::Instrument => {
                let identifier = InstrumentalVariableIdentifier {
                    config: InstrumentSearchConfig { max_results: MAX_CANDIDATES },
                };
                let prepared = identifier.prepare(&failure.dag).map_err(|e| e.to_string())?;
                let result = identifier
                    .identify(&prepared, &query, &mut workspace)
                    .map_err(|e| e.to_string())?;
                let sets = result
                    .estimands
                    .iter()
                    .filter_map(|e| e.instruments.first().copied())
                    .filter(|z| {
                        verify_instrument(&failure.dag, failure.treatment, failure.outcome, *z)
                    })
                    .map(|z| vec![z])
                    .collect();
                Ok((sets, false))
            }
        }
    }

    fn analyze(self, failure: &Failure, law: &BTreeSet<VariableId>) -> Analysis {
        let (sets, bounded) = match self.candidates(failure) {
            Ok(found) => found,
            Err(error) => return Analysis::Failed(error),
        };
        match sets.into_iter().find(|set| set.iter().all(|v| law.contains(v))) {
            Some(set) => Analysis::Identified(set),
            None if bounded => Analysis::Failed(format!(
                "no {} lies in the observed law and the search was bounded",
                self.witness()
            )),
            None => Analysis::NotIdentified(format!(
                "no {} of the graph lies in the observed law",
                self.witness()
            )),
        }
    }

    /// The evidence the contract waits for, read from the whole graph.
    #[allow(
        clippy::too_many_lines,
        reason = "one explicit obligation list per route, each with its own proof step"
    )]
    fn derive(self, failure: &Failure) -> Result<Vec<EvidenceObligation>, RepairError> {
        let source = failure.contract(self.name());
        let build = Build { family: self.name(), source: &source };
        let (sets, bounded) = self.candidates(failure).map_err(invalid)?;
        let mediator_bound = max_mediator_set_size(failure.dag.node_count());
        let missing =
            |set: &Vec<VariableId>| set.iter().filter(|v| !failure.base_law.contains(*v)).count();
        let best = sets.into_iter().min_by_key(|set| {
            let mut ids: Vec<u32> = set.iter().map(|v| v.raw()).collect();
            ids.sort_unstable();
            (missing(set), set.len(), ids)
        });
        let mut out = Vec::new();
        match (self, best) {
            (Self::FrontDoor, Some(set)) => {
                let mut needed: BTreeSet<VariableId> = set.iter().copied().collect();
                needed.insert(failure.treatment);
                needed.insert(failure.outcome);
                let list = id_list(&set);
                out.push(build.joint_law(
                    &needed,
                    &failure.population,
                    format!(
                        "front-door adjustment needs one joint observational law over the \
                         treatment, the outcome and a mediator set that intercepts every \
                         directed path and is shielded from the treatment's confounding \
                         (mediators {list})"
                    ),
                    "frontdoor:mediator_set",
                    format!("frontdoor.mediator_set:{list}"),
                )?);
            }
            (Self::Instrument, Some(set)) => {
                let z = set[0];
                let mut needed: BTreeSet<VariableId> = BTreeSet::new();
                needed.extend([z, failure.treatment, failure.outcome]);
                out.push(build.joint_law(
                    &needed,
                    &failure.population,
                    format!(
                        "the Wald ratio is read from one joint observational law over the \
                         instrument {}, the treatment and the outcome",
                        z.raw()
                    ),
                    "iv:joint_law",
                    format!("iv.instrument:{}", z.raw()),
                )?);
                out.push(build.assumption(
                    format!(
                        "instrument {} affects the outcome only through the treatment and shares \
                         no unmeasured cause with it (exclusion); data cannot establish this",
                        z.raw()
                    ),
                    format!("iv:exclusion:{}", z.raw()),
                    format!("iv.exclusion:{}", z.raw()),
                )?);
                out.push(build.assumption(
                    format!(
                        "instrument {} has a non-zero first-stage association with the treatment \
                         (relevance)",
                        z.raw()
                    ),
                    format!("iv:relevance:{}", z.raw()),
                    format!("iv.relevance:{}", z.raw()),
                )?);
                out.push(build.assumption(
                    "the Wald ratio is the average effect only under a constant linear effect, \
                     or the complier local effect under monotonicity"
                        .to_owned(),
                    "iv:effect_restriction".to_owned(),
                    "iv.effect_restriction".to_owned(),
                )?);
            }
            (Self::FrontDoor, None) => out.push(build.assumption(
                format!(
                    "no front-door mediator set was found in the graph{}",
                    if bounded {
                        format!(" within the search bound (sets of at most {mediator_bound})")
                    } else {
                        String::new()
                    }
                ),
                "frontdoor:no_mediator_set".to_owned(),
                "frontdoor.existence".to_owned(),
            )?),
            (Self::Instrument, None) => out.push(
                build.assumption(
                    "no variable of the graph satisfies the instrument conditions; an instrument \
                 outside the declared graph (a randomized encouragement, a natural source of \
                 variation) must be found and argued from design knowledge"
                        .to_owned(),
                    "iv:no_instrument".to_owned(),
                    "iv.existence".to_owned(),
                )?,
            ),
        }
        Ok(out)
    }
}

#[derive(Clone, Debug)]
struct Core {
    failure: Failure,
    route: Route,
    obligations: Vec<EvidenceObligation>,
}

impl Core {
    fn try_new(
        route: Route,
        dag: Dag,
        treatment: VariableId,
        outcome: VariableId,
        population: impl Into<Arc<str>>,
        base_law: impl IntoIterator<Item = VariableId>,
    ) -> Result<Self, RepairError> {
        let failure = Failure::try_new(dag, treatment, outcome, population, base_law)?;
        // An instrument's validity (relevance, exclusion, the effect restriction) is an
        // assumption no observed law establishes, so a measured instrument never makes the
        // route "already identified": the route keeps owing those assumptions.
        if route != Route::Instrument
            && failure.usable(&failure.base_law)
            && matches!(route.analyze(&failure, &failure.base_law), Analysis::Identified(_))
        {
            return Err(RepairError::not_a_failure(format!(
                "the base law already identifies the effect by the {} route",
                route.name()
            )));
        }
        let obligations = route.derive(&failure)?;
        Ok(Self { failure, route, obligations })
    }

    fn decide(&self, laws: &[(String, BTreeSet<VariableId>)]) -> FamilyVerdict {
        let mut reasons = Vec::new();
        for (name, law) in laws {
            match self.route.analyze(&self.failure, law) {
                Analysis::Identified(set) => {
                    return FamilyVerdict::Sufficient(RepairDerivation {
                        checker: self.route.checker(),
                        steps: vec![
                            format!("law:{name}"),
                            format!("{}:[{}]", self.route.witness(), id_list(&set)),
                        ],
                        verified: true,
                    });
                }
                Analysis::NotIdentified(why) => reasons.push(format!("law {name}: {why}")),
                Analysis::Failed(why) => {
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

    fn check(
        &self,
        studies: &[&DurableStudyCandidate],
        ctx: &ExecutionContext,
    ) -> Result<FamilyVerdict, RepairCancelled> {
        if ctx.cancellation.is_cancelled() {
            return Err(RepairCancelled);
        }
        Ok(match self.failure.laws_of_studies(studies) {
            Ok(laws) => self.decide(&laws),
            Err(reasons) => FamilyVerdict::Invalid { reasons },
        })
    }

    fn check_regimes(
        &self,
        regimes: &[EvidenceRegime],
        ctx: &ExecutionContext,
    ) -> Result<FamilyVerdict, RepairCancelled> {
        if ctx.cancellation.is_cancelled() {
            return Err(RepairCancelled);
        }
        Ok(self.decide(&self.failure.laws_of_regimes(regimes)))
    }
}

macro_rules! graph_route_family {
    ($(#[$meta:meta])* $name:ident, $route:expr, $id:literal) => {
        $(#[$meta])*
        #[derive(Clone, Debug)]
        pub struct $name {
            core: Core,
        }

        impl $name {
            /// Freeze a failed contract: the variables of `base_law` are those
            /// currently observed jointly (in `population`) with the treatment
            /// and the outcome, and the graph may hold variables nobody measures.
            ///
            /// # Errors
            /// `identification_repair.invalid_request` for an unknown or equal
            /// treatment/outcome or an empty population;
            /// `identification_repair.not_a_failure` when the base law already
            /// identifies the effect by this route.
            pub fn try_new(
                dag: Dag,
                treatment: VariableId,
                outcome: VariableId,
                population: impl Into<Arc<str>>,
                base_law: impl IntoIterator<Item = VariableId>,
            ) -> Result<Self, RepairError> {
                Core::try_new($route, dag, treatment, outcome, population, base_law)
                    .map(|core| Self { core })
            }

            /// The failed contract's graph.
            #[must_use]
            pub fn dag(&self) -> &Dag {
                &self.core.failure.dag
            }

            /// Re-run the route's identifier on already-proposed `regimes` (the
            /// stored hypothetical delta of an artifact), exactly as
            /// [`RepairFamily::check`] reads a study's expected evidence.
            ///
            /// # Errors
            /// Cancellation.
            pub fn check_regimes(
                &self,
                regimes: &[EvidenceRegime],
                ctx: &ExecutionContext,
            ) -> Result<FamilyVerdict, RepairCancelled> {
                self.core.check_regimes(regimes, ctx)
            }
        }

        impl FailedContract for $name {
            fn treatment(&self) -> VariableId {
                self.core.failure.treatment
            }

            fn outcome(&self) -> VariableId {
                self.core.failure.outcome
            }

            fn population(&self) -> &str {
                &self.core.failure.population
            }

            fn base_law(&self) -> Vec<VariableId> {
                self.core.failure.base_law.iter().copied().collect()
            }
        }

        impl RepairFamily for $name {
            fn family_id(&self) -> &'static str {
                $id
            }

            fn contract_id(&self) -> String {
                self.core.failure.contract($id)
            }

            fn unresolved_obligations(&self) -> Vec<EvidenceObligation> {
                self.core.obligations.clone()
            }

            fn hypothetical_regimes(
                &self,
                studies: &[&DurableStudyCandidate],
            ) -> Result<Vec<EvidenceRegime>, Vec<String>> {
                regimes_of(studies)
            }

            fn check(
                &self,
                studies: &[&DurableStudyCandidate],
                ctx: &ExecutionContext,
            ) -> Result<FamilyVerdict, RepairCancelled> {
                self.core.check(studies, ctx)
            }
        }
    };
}

graph_route_family!(
    /// Front-door adjustment as a repair family.
    ///
    /// The failed contract is "identify the average effect of the treatment on
    /// the outcome through a front-door mediator set read from one joint
    /// observational law". A study repairs it by delivering that joint law over
    /// the treatment, the outcome and a mediator set the criterion accepts; a
    /// mediator measured in a separate study never combines.
    FrontDoorRepairFamily,
    Route::FrontDoor,
    "frontdoor"
);

graph_route_family!(
    /// Instrumental-variable identification as a repair family.
    ///
    /// A study can deliver the joint observational law of an instrument, the
    /// treatment and the outcome, but relevance, exclusion and the effect
    /// restriction are `EstablishAssumption` obligations that no study satisfies,
    /// so a report over this family never certifies a repair on data alone.
    InstrumentRepairFamily,
    Route::Instrument,
    "instrument"
);

// ------------------------------------------------------------------- hedge

// ------------------------------------------------------------- sensitivity

#[cfg(test)]
mod tests;

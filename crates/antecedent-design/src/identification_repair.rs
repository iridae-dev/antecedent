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

use crate::repair::{
    BackdoorRepairFamily, FamilyVerdict, RepairCancelled, RepairDerivation, RepairError,
    RepairFamily, dense_of, graph_variables,
};
use crate::study_candidate::DurableStudyCandidate;

/// Most candidate mediator sets or instruments one search collects.
const MAX_CANDIDATES: usize = 4096;
/// Largest mediator set the front-door search enumerates.
const MAX_MEDIATOR_SET: usize = 4;

fn id_list(variables: &[VariableId]) -> String {
    let mut ids: Vec<u32> = variables.iter().map(|v| v.raw()).collect();
    ids.sort_unstable();
    ids.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
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
                        max_mediator_set_size: MAX_MEDIATOR_SET,
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
                        format!(" within the search bound (sets of at most {MAX_MEDIATOR_SET})")
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
        if failure.usable(&failure.base_law)
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

            /// The treatment variable.
            #[must_use]
            pub const fn treatment(&self) -> VariableId {
                self.core.failure.treatment
            }

            /// The outcome variable.
            #[must_use]
            pub const fn outcome(&self) -> VariableId {
                self.core.failure.outcome
            }

            /// The contract's population.
            #[must_use]
            pub fn population(&self) -> &str {
                &self.core.failure.population
            }

            /// The variables currently observed jointly with treatment and outcome.
            #[must_use]
            pub fn base_law(&self) -> Vec<VariableId> {
                self.core.failure.base_law.iter().copied().collect()
            }

            /// The proposed regimes `studies` would deliver, with consecutive
            /// ids from zero.
            ///
            /// # Errors
            /// The failure reasons of an invalid study.
            pub fn hypothetical_regimes(
                &self,
                studies: &[&DurableStudyCandidate],
            ) -> Result<Vec<EvidenceRegime>, Vec<String>> {
                regimes_of(studies)
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

/// Experiments that answer a hedge.
///
/// A hedge is a proof that `P(y | do(x))` is not a functional of the observed
/// law. The family accepts only experiments: setting the treatment and measuring
/// the outcome observes the interventional law directly, and setting one
/// verified surrogate node (one that intercepts every directed path from
/// treatment to outcome while `P(w | do(x))` is identified from observation, so
/// `P(y | do(x)) = sum_w P(w | do(x)) P(y | do(w))`) identifies it. The
/// surrogate route also needs the observational law over every graph variable.
#[derive(Clone, Debug)]
pub struct HedgeRepairFamily {
    admg: Admg,
    treatment: VariableId,
    outcome: VariableId,
    population: Arc<str>,
    base_law: BTreeSet<VariableId>,
    certificate: Option<HedgeCertificate>,
    surrogates: Vec<VariableId>,
    obligations: Vec<EvidenceObligation>,
}

fn hedge_identifies(
    prepared: &antecedent_identify::PreparedAdmg,
    treatment: VariableId,
    outcome: VariableId,
) -> Result<antecedent_identify::IdentificationResult, RepairError> {
    IdIdentifier::new()
        .identify_ate(
            prepared,
            &AverageEffectQuery::binary_ate(treatment, outcome),
            &mut IdentificationWorkspace::default(),
        )
        .map_err(|e| RepairError::from_identification(&e))
}

impl HedgeRepairFamily {
    /// Freeze a failed contract in an ADMG: `base_law` are the variables
    /// observed jointly in `population`.
    ///
    /// # Errors
    /// `identification_repair.invalid_request` for an unknown or equal
    /// treatment/outcome, a non-static graph or an empty population;
    /// `identification_repair.not_a_failure` when the ID algorithm identifies the
    /// effect; `identification_repair.budget` when it stops.
    pub fn try_new(
        admg: Admg,
        treatment: VariableId,
        outcome: VariableId,
        population: impl Into<Arc<str>>,
        base_law: impl IntoIterator<Item = VariableId>,
    ) -> Result<Self, RepairError> {
        let population = population.into();
        let mut variables = Vec::new();
        for node in admg.nodes() {
            match node {
                NodeRef::Static(variable) => variables.push(*variable),
                _ => return Err(invalid("a failure graph holds only static variables")),
            }
        }
        let base_law: BTreeSet<VariableId> = base_law.into_iter().collect();
        if treatment == outcome
            || !variables.contains(&treatment)
            || !variables.contains(&outcome)
            || base_law.iter().any(|v| !variables.contains(v))
            || is_blank(&population)
        {
            return Err(invalid(
                "a hedge contract names a population and a treatment and outcome that differ and \
                 lie in the graph",
            ));
        }
        let prepared = IdIdentifier::new().prepare(&admg).map_err(|e| invalid(e.to_string()))?;
        let result = hedge_identifies(&prepared, treatment, outcome)?;
        if result.status == IdentificationStatus::NonparametricallyIdentified {
            return Err(RepairError::not_a_failure("the ID algorithm identifies the effect"));
        }
        let certificate =
            result.hedge.filter(|cert| cert.verify(&prepared, &[treatment], &[outcome]).is_ok());
        let mut family = Self {
            admg,
            treatment,
            outcome,
            population,
            base_law,
            certificate,
            surrogates: Vec::new(),
            obligations: Vec::new(),
        };
        family.surrogates = family.verified_surrogates(&prepared);
        family.obligations = family.derive_obligations()?;
        Ok(family)
    }

    /// Hedge nodes (other than treatment and outcome) whose experiment identifies
    /// the effect.
    fn verified_surrogates(&self, prepared: &antecedent_identify::PreparedAdmg) -> Vec<VariableId> {
        let Some(cert) = &self.certificate else { return Vec::new() };
        let mut nodes: BTreeSet<VariableId> =
            cert.f.iter().chain(cert.f_prime.iter()).copied().collect();
        nodes.remove(&self.treatment);
        nodes.remove(&self.outcome);
        nodes.into_iter().filter(|w| self.surrogate_resolves(prepared, *w)).collect()
    }

    fn surrogate_resolves(
        &self,
        prepared: &antecedent_identify::PreparedAdmg,
        w: VariableId,
    ) -> bool {
        let (Some(t), Some(y), Some(wd)) = (
            self.admg.dense_of_static(self.treatment),
            self.admg.dense_of_static(self.outcome),
            self.admg.dense_of_static(w),
        ) else {
            return false;
        };
        let children = |node: DenseNodeId| self.admg.children(node).to_vec();
        !reaches_avoiding(self.admg.node_count(), &children, t, y, &[wd])
            && hedge_identifies(prepared, self.treatment, w)
                .is_ok_and(|r| r.status == IdentificationStatus::NonparametricallyIdentified)
    }

    fn derive_obligations(&self) -> Result<Vec<EvidenceObligation>, RepairError> {
        let source = self.contract_id();
        let build = Build { family: "hedge", source: &source };
        let (why, step) = match &self.certificate {
            Some(cert) => (
                format!(
                    "observation cannot identify the effect: the graph holds a hedge (F=[{}], \
                     F'=[{}])",
                    id_list(&cert.f),
                    id_list(&cert.f_prime)
                ),
                format!("hedge:f=[{}];f_prime=[{}]", id_list(&cert.f), id_list(&cert.f_prime)),
            ),
            None => (
                "the ID algorithm does not identify the effect from observation".to_owned(),
                "id.not_identified".to_owned(),
            ),
        };
        let mut out = vec![build.experiment(
            self.treatment,
            self.outcome,
            &self.population,
            format!(
                "{why}; an experiment that sets treatment {} and measures outcome {} observes \
                 P(y | do(x)) directly",
                self.treatment.raw(),
                self.outcome.raw()
            ),
            format!("hedge:experiment:{}", self.treatment.raw()),
            step,
        )?];
        for w in &self.surrogates {
            out.push(build.experiment(
                *w,
                self.outcome,
                &self.population,
                format!(
                    "{why}; node {} lies on the hedge and intercepts every directed path from \
                     the treatment to the outcome, and P({} | do(treatment)) is identified from \
                     observation, so an experiment on {} with outcome {} identifies \
                     P(y | do(x)) = sum_w P(w | do(x)) P(y | do(w))",
                    w.raw(),
                    w.raw(),
                    w.raw(),
                    self.outcome.raw()
                ),
                format!("hedge:surrogate:{}", w.raw()),
                format!("hedge.surrogate:{}", w.raw()),
            )?);
        }
        Ok(out)
    }

    /// The failed contract's graph.
    #[must_use]
    pub fn admg(&self) -> &Admg {
        &self.admg
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

    /// The variables currently observed jointly.
    #[must_use]
    pub fn base_law(&self) -> Vec<VariableId> {
        self.base_law.iter().copied().collect()
    }

    /// The hedge witness the ID algorithm produced and the verifier confirmed.
    #[must_use]
    pub fn certificate(&self) -> Option<&HedgeCertificate> {
        self.certificate.as_ref()
    }

    /// The hedge nodes whose experiment identifies the effect.
    #[must_use]
    pub fn surrogates(&self) -> &[VariableId] {
        &self.surrogates
    }

    /// The proposed regimes `studies` would deliver, with consecutive ids from zero.
    ///
    /// # Errors
    /// The failure reasons of an invalid study.
    pub fn hypothetical_regimes(
        &self,
        studies: &[&DurableStudyCandidate],
    ) -> Result<Vec<EvidenceRegime>, Vec<String>> {
        regimes_of(studies)
    }

    /// The verdict over the single-target experiments on offer, as
    /// `(label, target)` pairs.
    fn decide(&self, experiments: &[(String, VariableId)]) -> FamilyVerdict {
        let mut reasons = Vec::new();
        for (label, target) in experiments {
            if *target == self.treatment {
                return FamilyVerdict::Sufficient(RepairDerivation {
                    checker: "hedge.experiment",
                    steps: vec![format!("experiment:{label}"), "direct:P(y | do(x))".to_owned()],
                    verified: true,
                });
            }
            if self.surrogates.contains(target) {
                let covered =
                    graph_variables_of(&self.admg).iter().all(|v| self.base_law.contains(v));
                if covered {
                    return FamilyVerdict::Sufficient(RepairDerivation {
                        checker: "hedge.experiment",
                        steps: vec![
                            format!("experiment:{label}"),
                            format!("surrogate:{}", target.raw()),
                            "P(y | do(x)) = sum_w P(w | do(x)) P(y | do(w))".to_owned(),
                        ],
                        verified: true,
                    });
                }
                reasons.push(format!(
                    "experiment {label} on surrogate {} needs the observational joint law over \
                     every graph variable to identify P(w | do(x))",
                    target.raw()
                ));
            } else {
                reasons.push(format!(
                    "experiment {label} on {} neither sets the treatment nor is a verified \
                     surrogate on the hedge",
                    target.raw()
                ));
            }
        }
        if experiments.is_empty() {
            reasons.push(
                "no experiment in the contract's population measures the outcome while setting \
                 one variable"
                    .to_owned(),
            );
        }
        FamilyVerdict::Insufficient { reasons }
    }

    /// Re-run the verdict on already-proposed `regimes`.
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
        let experiments: Vec<(String, VariableId)> = regimes
            .iter()
            .filter(|regime| {
                regime.population == self.population
                    && regime.interventions.len() == 1
                    && regime.conditioned_on.is_empty()
                    && regime.measured.contains(&self.outcome)
                    && outcome_available(&regime.distribution, self.outcome)
            })
            .map(|regime| (format!("regime:{}", regime.id.raw()), regime.interventions[0]))
            .collect();
        Ok(self.decide(&experiments))
    }
}

fn graph_variables_of(admg: &Admg) -> Vec<VariableId> {
    admg.nodes()
        .iter()
        .filter_map(|node| match node {
            NodeRef::Static(variable) => Some(*variable),
            _ => None,
        })
        .collect()
}

/// Whether the outcome's marginal is part of the law (a joint law, or a
/// separate marginal that names it).
fn outcome_available(distribution: &DistributionAvailability, outcome: VariableId) -> bool {
    match distribution {
        DistributionAvailability::Joint => true,
        DistributionAvailability::SeparateMarginals { variables } => variables.contains(&outcome),
    }
}

impl RepairFamily for HedgeRepairFamily {
    fn family_id(&self) -> &'static str {
        "hedge"
    }

    fn contract_id(&self) -> String {
        format!(
            "hedge:{}:treatment={}:outcome={}",
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
        let mut experiments = Vec::new();
        let mut malformed = Vec::new();
        for study in studies {
            if let Err(error) = study.validate() {
                malformed.push(error.to_string());
                continue;
            }
            for (k, evidence) in study.expected_evidence.iter().enumerate() {
                if evidence.population == self.population
                    && evidence.interventions.len() == 1
                    && evidence.conditioned_on.is_empty()
                    && evidence.measured.contains(&self.outcome)
                    && outcome_available(&evidence.distribution, self.outcome)
                {
                    experiments
                        .push((format!("{}#{k}", study.semantic_id()), evidence.interventions[0]));
                }
            }
        }
        if !malformed.is_empty() {
            return Ok(FamilyVerdict::Invalid { reasons: malformed });
        }
        Ok(self.decide(&experiments))
    }
}

// ------------------------------------------------------------- sensitivity

/// The sensitivity analysis a finding came from.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum SensitivityMethod {
    /// E-value: the risk-ratio strength of a confounder that explains the effect away.
    EValue,
    /// Omitted-variable bias: the partial R-squared of a confounder that does.
    OmittedVariableBias,
    /// A tipping search over the declared confounding parameter.
    Tipping,
}

impl SensitivityMethod {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EValue => "e_value",
            Self::OmittedVariableBias => "omitted_variable_bias",
            Self::Tipping => "tipping",
        }
    }

    /// The method named `name`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::EValue, Self::OmittedVariableBias, Self::Tipping]
            .into_iter()
            .find(|method| method.as_str() == name)
    }
}

/// What a sensitivity analysis found about unmeasured confounding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SensitivityFinding {
    /// The analysis.
    pub method: SensitivityMethod,
    /// The confounder strength at which the conclusion is overturned: the
    /// e-value, the robustness value or the tipping parameter.
    pub strength: f64,
    /// Whether the conclusion survives the confounding the analyst deems plausible.
    /// A robust conclusion owes no evidence.
    pub robust: bool,
}

/// A sensitivity finding as a repair family.
///
/// The conclusion is overturned by an unmeasured confounder `C` of the reported
/// strength. The family models `C` as one more common cause of the treatment and
/// the outcome and is repaired by a study that measures `C` jointly with the
/// treatment, the outcome and whatever else the back-door criterion then needs.
#[derive(Clone, Debug)]
pub struct SensitivityRepairFamily {
    base: Dag,
    finding: SensitivityFinding,
    confounder: VariableId,
    inner: BackdoorRepairFamily,
    obligations: Vec<EvidenceObligation>,
}

impl SensitivityRepairFamily {
    /// Freeze a non-robust sensitivity finding over `dag` (latent causes
    /// included); `base_law` are the variables observed jointly.
    ///
    /// # Errors
    /// `identification_repair.invalid_request` for an unknown or equal
    /// treatment/outcome, an empty population or a non-positive strength;
    /// `identification_repair.not_a_failure` when the finding is robust.
    pub fn try_new(
        dag: Dag,
        treatment: VariableId,
        outcome: VariableId,
        population: impl Into<Arc<str>>,
        base_law: impl IntoIterator<Item = VariableId>,
        finding: SensitivityFinding,
    ) -> Result<Self, RepairError> {
        if !(finding.strength.is_finite() && finding.strength > 0.0) {
            return Err(invalid("a sensitivity strength is finite and positive"));
        }
        if finding.robust {
            return Err(RepairError::not_a_failure(
                "the conclusion survives the plausible unmeasured confounding",
            ));
        }
        let population = population.into();
        let base_law: BTreeSet<VariableId> = base_law.into_iter().collect();
        // Validates the treatment, outcome, population and base law.
        Failure::try_new(
            dag.clone(),
            treatment,
            outcome,
            Arc::clone(&population),
            base_law.clone(),
        )?;
        let confounder = VariableId::from_raw(
            graph_variables(&dag).iter().map(|v| v.raw()).max().map_or(0, |m| m.saturating_add(1)),
        );
        let mut augmented = dag.clone();
        let node =
            augmented.add_node(NodeRef::Static(confounder)).map_err(|e| invalid(e.to_string()))?;
        for end in [treatment, outcome] {
            let end = dense_of(&augmented, end).ok_or_else(|| invalid("unknown variable"))?;
            augmented.insert_directed(node, end).map_err(|e| invalid(e.to_string()))?;
        }
        let inner = BackdoorRepairFamily::try_new(
            augmented,
            treatment,
            outcome,
            Arc::clone(&population),
            base_law,
            &[],
        )?;
        let mut family = Self { base: dag, finding, confounder, inner, obligations: Vec::new() };
        family.obligations = family.restate()?;
        Ok(family)
    }

    /// The back-door obligations of the confounded graph, restated with the
    /// sensitivity finding as their provenance.
    fn restate(&self) -> Result<Vec<EvidenceObligation>, RepairError> {
        let source = self.contract_id();
        let step = format!(
            "sensitivity.{}:strength={}",
            self.finding.method.as_str(),
            self.finding.strength
        );
        self.inner
            .unresolved_obligations()
            .into_iter()
            .map(|o| {
                EvidenceObligation::try_new(EvidenceObligationSpec {
                    quantities: o.quantities,
                    kind: o.kind,
                    scope: o.scope,
                    variables: o.variables,
                    population: o.population,
                    regime: o.regime,
                    reason: Arc::from(format!(
                        "an unmeasured confounder of strength {} ({}) would overturn the \
                         conclusion; {}",
                        self.finding.strength,
                        self.finding.method.as_str(),
                        o.reason
                    )),
                    required_slots: o.required_slots,
                    min_additional_samples: o.min_additional_samples,
                    provenance: ObligationProvenance {
                        family: Arc::from("sensitivity"),
                        source: Arc::from(source.as_str()),
                        proof_step: Some(Arc::from(step.as_str())),
                    },
                })
                .map_err(|e| invalid(e.to_string()))
            })
            .collect()
    }

    /// The graph the finding was reported on (without the modelled confounder).
    #[must_use]
    pub fn dag(&self) -> &Dag {
        &self.base
    }

    /// The treatment variable.
    #[must_use]
    pub const fn treatment(&self) -> VariableId {
        self.inner.treatment()
    }

    /// The outcome variable.
    #[must_use]
    pub const fn outcome(&self) -> VariableId {
        self.inner.outcome()
    }

    /// The contract's population.
    #[must_use]
    pub fn population(&self) -> &str {
        self.inner.population()
    }

    /// The variables currently observed jointly.
    #[must_use]
    pub fn base_law(&self) -> Vec<VariableId> {
        self.inner.base_law()
    }

    /// The finding.
    #[must_use]
    pub const fn finding(&self) -> SensitivityFinding {
        self.finding
    }

    /// The variable standing for the unmeasured confounder.
    #[must_use]
    pub const fn confounder(&self) -> VariableId {
        self.confounder
    }

    /// The proposed regimes `studies` would deliver, with consecutive ids from zero.
    ///
    /// # Errors
    /// The failure reasons of an invalid study.
    pub fn hypothetical_regimes(
        &self,
        studies: &[&DurableStudyCandidate],
    ) -> Result<Vec<EvidenceRegime>, Vec<String>> {
        self.inner.hypothetical_regimes(studies)
    }

    fn relabel(&self, verdict: FamilyVerdict) -> FamilyVerdict {
        match verdict {
            FamilyVerdict::Sufficient(mut derivation) => {
                derivation.checker = "sensitivity.backdoor_adjustment";
                derivation.steps.push(format!("confounder:{}", self.confounder.raw()));
                FamilyVerdict::Sufficient(derivation)
            }
            other => other,
        }
    }

    /// Re-run the back-door identifier on already-proposed `regimes`.
    ///
    /// # Errors
    /// Cancellation.
    pub fn check_regimes(
        &self,
        regimes: &[EvidenceRegime],
        ctx: &ExecutionContext,
    ) -> Result<FamilyVerdict, RepairCancelled> {
        self.inner.check_regimes(regimes, ctx).map(|verdict| self.relabel(verdict))
    }
}

impl RepairFamily for SensitivityRepairFamily {
    fn family_id(&self) -> &'static str {
        "sensitivity"
    }

    fn contract_id(&self) -> String {
        format!(
            "sensitivity:{}:treatment={}:outcome={}:method={}",
            self.inner.population(),
            self.inner.treatment().raw(),
            self.inner.outcome().raw(),
            self.finding.method.as_str()
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
        self.inner.check(studies, ctx).map(|verdict| self.relabel(verdict))
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{AssumptionSet, SearchLimits};
    use antecedent_identify::{
        DerivationTrace, IdentificationPerformanceRecord, IdentificationResult,
    };

    use super::*;
    use crate::obligation_adapters::identification_failure_obligations;
    use crate::repair::{RepairClassification, RepairLimits, RepairObjective, repair};
    use crate::study_candidate::{ExpectedEvidence, StudyCostDeclaration, StudyKind, UnitRules};

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn dag(nodes: u32, edges: &[(u32, u32)]) -> Dag {
        let mut dag = Dag::with_variables(nodes);
        for (a, b) in edges {
            dag.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
        }
        dag
    }

    fn admg(nodes: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
        let mut graph = Admg::with_variables(nodes);
        for (a, b) in directed {
            graph.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
        }
        for (a, b) in bidirected {
            graph.insert_bidirected(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
        }
        graph
    }

    fn failed(treatment: u32, outcome: u32) -> IdentificationResult {
        IdentificationResult::not_identified(
            CausalQuery::average_effect(AverageEffectQuery::binary_ate(v(treatment), v(outcome))),
            DerivationTrace::default(),
            AssumptionSet::new(),
            IdentificationPerformanceRecord::default(),
        )
    }

    fn steps(obligations: &[EvidenceObligation], family: &str) -> Vec<String> {
        obligations
            .iter()
            .filter(|o| o.provenance.family.as_ref() == family)
            .filter_map(|o| o.provenance.proof_step.as_ref().map(ToString::to_string))
            .collect()
    }

    fn candidate(
        kind: StudyKind,
        interventions: &[u32],
        measured: &[u32],
        label: &str,
    ) -> DurableStudyCandidate {
        let ids = |raw: &[u32]| raw.iter().copied().map(v).collect::<Arc<[VariableId]>>();
        DurableStudyCandidate {
            label: Arc::from(label),
            kind,
            population: Arc::from("pop"),
            interventions: ids(interventions),
            measured: ids(measured),
            joint_measurement: true,
            sample_size: 100,
            recruitment: Arc::from("census"),
            timing: Arc::from("one wave"),
            unit_rules: UnitRules {
                unit: Arc::from("patient"),
                cluster: None,
                whole_cluster_sampling: false,
            },
            cost: StudyCostDeclaration {
                units: 5,
                unit_label: Arc::from("usd"),
                sample_budget: 100,
            },
            feasible: true,
            feasibility_notes: Arc::from([]),
            expected_evidence: Arc::from([ExpectedEvidence {
                population: Arc::from("pop"),
                interventions: ids(interventions),
                intervention_values: Arc::from([]),
                conditioned_on: Arc::from([]),
                measured: ids(measured),
                distribution: DistributionAvailability::Joint,
            }]),
            external_provider: None,
        }
    }

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(1)
    }

    fn limits() -> RepairLimits {
        RepairLimits {
            search: SearchLimits { operations: 1000, depth: 2 },
            memory_limit_bytes: 16 * 1024 * 1024,
        }
    }

    /// T=0 -> Y=1 confounded by the unmeasured U=2.
    #[test]
    fn a_confounded_graph_owes_the_back_door_set_with_its_proof_step() {
        let graph = Admg::from_dag(&dag(3, &[(0, 1), (2, 0), (2, 1)])).unwrap();
        let obligations =
            identification_failure_obligations(&failed(0, 1), &graph, &[v(0), v(1)], "pop", None)
                .unwrap();
        assert_eq!(steps(&obligations, "backdoor"), vec!["backdoor.adjustment_set:0,1,2"]);
        let joint =
            obligations.iter().find(|o| o.provenance.family.as_ref() == "backdoor").unwrap();
        assert_eq!(joint.kind, EvidenceObligationKind::ProvideJointLaw);
        assert!(joint.variables.contains(&v(2)));
        // Unmeasured confounding also owes the panel alternative, with its assumption.
        assert!(obligations.iter().any(|o| o.provenance.family.as_ref() == "panel"));
    }

    /// T=0 -> M=1 -> Y=2 with U=3 confounding T and Y; only T and Y are observed.
    #[test]
    fn an_unmeasured_front_door_mediator_is_owed_as_a_joint_law() {
        let graph = Admg::from_dag(&dag(4, &[(0, 1), (1, 2), (3, 0), (3, 2)])).unwrap();
        let obligations =
            identification_failure_obligations(&failed(0, 2), &graph, &[v(0), v(2)], "pop", None)
                .unwrap();
        assert_eq!(steps(&obligations, "frontdoor"), vec!["frontdoor.mediator_set:1"]);
        let front =
            obligations.iter().find(|o| o.provenance.family.as_ref() == "frontdoor").unwrap();
        assert_eq!(front.kind, EvidenceObligationKind::ProvideJointLaw);
        assert_eq!(front.variables.iter().map(|x| x.raw()).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn a_base_law_that_already_identifies_refuses() {
        let graph = dag(4, &[(0, 1), (1, 2), (3, 0), (3, 2)]);
        let observed = [v(0), v(1), v(2)];
        let refused =
            FrontDoorRepairFamily::try_new(graph.clone(), v(0), v(2), "pop", observed).unwrap_err();
        assert_eq!(refused.detail, "identification_repair.not_a_failure");
        // Back-door adjustment over everything observed, with U measured as well.
        let all = [v(0), v(1), v(2), v(3)];
        let refused =
            BackdoorRepairFamily::try_new(graph, v(0), v(2), "pop", all, &[]).unwrap_err();
        assert_eq!(refused.detail, "identification_repair.not_a_failure");
        // An identified result owes nothing.
        let mut identified = failed(0, 1);
        identified.status = IdentificationStatus::NonparametricallyIdentified;
        let graph = Admg::from_dag(&dag(2, &[(0, 1)])).unwrap();
        let refused =
            identification_failure_obligations(&identified, &graph, &[v(0), v(1)], "pop", None)
                .unwrap_err();
        assert_eq!(refused.detail, "identification_repair.not_a_failure");
    }

    /// Z=0 -> T=1 -> Y=2 with U=3 confounding T and Y; Z is not observed.
    #[test]
    fn an_instrument_is_owed_with_its_validity_recorded_as_assumptions() {
        let graph = dag(4, &[(0, 1), (1, 2), (3, 1), (3, 2)]);
        let family =
            InstrumentRepairFamily::try_new(graph, v(1), v(2), "pop", [v(1), v(2)]).unwrap();
        let obligations = family.unresolved_obligations();
        let steps: Vec<_> =
            obligations.iter().filter_map(|o| o.provenance.proof_step.as_deref()).collect();
        assert_eq!(
            steps,
            vec!["iv.instrument:0", "iv.exclusion:0", "iv.relevance:0", "iv.effect_restriction"]
        );
        let assumptions: Vec<_> = obligations
            .iter()
            .filter(|o| o.kind == EvidenceObligationKind::EstablishAssumption)
            .collect();
        assert_eq!(assumptions.len(), 3);
        assert!(assumptions.iter().all(|o| !o.satisfiable_by_study()));

        // A study that delivers the joint law is the data the Wald ratio needs, yet
        // no study certifies the family: validity stays an assumption.
        let study = candidate(StudyKind::Observation, &[], &[0, 1, 2], "instrumented survey");
        let report = repair(
            &family,
            std::slice::from_ref(&study),
            RepairObjective::MinimizeCost,
            limits(),
            &ctx(),
        )
        .unwrap();
        assert_eq!(report.outcomes[0].classification, RepairClassification::NotCertified);
        assert!(report.best().is_none());
        assert_eq!(
            report.status(),
            Some((
                antecedent_core::reason_code!("transport_missing_evidence"),
                "evidence_obligations.wrong_contract"
            ))
        );
    }

    #[test]
    fn a_graph_without_an_instrument_owes_the_search_for_one() {
        let graph = dag(3, &[(1, 2), (0, 1), (0, 2)]);
        let family =
            InstrumentRepairFamily::try_new(graph, v(1), v(2), "pop", [v(1), v(2)]).unwrap();
        let obligations = family.unresolved_obligations();
        assert_eq!(obligations.len(), 1);
        assert_eq!(obligations[0].kind, EvidenceObligationKind::EstablishAssumption);
        assert_eq!(obligations[0].provenance.proof_step.as_deref(), Some("iv.existence"));
    }

    /// T=0 -> W=1 -> Y=2 with T<->Y and W<->Y: a hedge on {T, W, Y}.
    #[test]
    fn a_hedge_owes_experiments_on_the_treatment_and_on_a_verified_surrogate() {
        let graph = admg(3, &[(0, 1), (1, 2)], &[(0, 2), (1, 2)]);
        let family =
            HedgeRepairFamily::try_new(graph.clone(), v(0), v(2), "pop", [v(0), v(1), v(2)])
                .unwrap();
        assert!(family.certificate().is_some());
        assert_eq!(family.surrogates(), &[v(1)]);
        let obligations = family.unresolved_obligations();
        assert!(obligations.iter().all(|o| o.kind == EvidenceObligationKind::Intervene));
        let targets: Vec<u32> =
            obligations.iter().map(|o| o.regime.interventions[0].raw()).collect();
        assert_eq!(targets, vec![0, 1]);
        assert!(obligations[0].provenance.proof_step.as_deref().unwrap().starts_with("hedge:f=["));
        assert_eq!(obligations[1].provenance.proof_step.as_deref(), Some("hedge.surrogate:1"));

        let direct = candidate(StudyKind::Experiment, &[0], &[2], "randomized trial");
        let on_w = candidate(StudyKind::Experiment, &[1], &[2], "mediator trial");
        let elsewhere = candidate(StudyKind::Experiment, &[2], &[0], "wrong direction");
        let report = repair(
            &family,
            &[direct, on_w, elsewhere],
            RepairObjective::MinimizeCost,
            limits(),
            &ctx(),
        )
        .unwrap();
        let verified: Vec<_> = report
            .outcomes
            .iter()
            .filter(|o| o.classification == RepairClassification::VerifiedSufficient)
            .collect();
        assert_eq!(verified.len(), 2);
        assert!(verified.iter().all(|o| o.candidates.len() == 1));
        // An observation never answers a hedge.
        let observed = candidate(StudyKind::Observation, &[], &[0, 1, 2], "bigger survey");
        let report =
            repair(&family, &[observed], RepairObjective::MinimizeCost, limits(), &ctx()).unwrap();
        assert!(report.best().is_none());
    }

    #[test]
    fn an_identified_hedge_contract_refuses() {
        let graph = admg(2, &[(0, 1)], &[]);
        let refused =
            HedgeRepairFamily::try_new(graph, v(0), v(1), "pop", [v(0), v(1)]).unwrap_err();
        assert_eq!(refused.detail, "identification_repair.not_a_failure");
    }

    #[test]
    fn a_sensitivity_finding_owes_the_confounder_and_a_robust_one_refuses() {
        // T=0 -> Y=1 adjusted for the measured Z=2; the analysis says a confounder
        // of e-value 1.4 would overturn the conclusion.
        let graph = dag(3, &[(0, 1), (2, 0), (2, 1)]);
        let finding =
            SensitivityFinding { method: SensitivityMethod::EValue, strength: 1.4, robust: false };
        let family = SensitivityRepairFamily::try_new(
            graph.clone(),
            v(0),
            v(1),
            "pop",
            [v(0), v(1), v(2)],
            finding,
        )
        .unwrap();
        assert_eq!(family.confounder(), v(3));
        let obligations = family.unresolved_obligations();
        assert_eq!(obligations.len(), 1);
        assert_eq!(obligations[0].provenance.family.as_ref(), "sensitivity");
        assert_eq!(
            obligations[0].provenance.proof_step.as_deref(),
            Some("sensitivity.e_value:strength=1.4")
        );
        assert!(obligations[0].variables.contains(&v(3)));
        // Measuring the confounder jointly repairs it; the measured covariates alone do not.
        let measured = candidate(StudyKind::Observation, &[], &[0, 1, 2, 3], "assay");
        let unchanged = candidate(StudyKind::Observation, &[], &[0, 1, 2], "more rows");
        let report = repair(
            &family,
            &[measured, unchanged],
            RepairObjective::MinimizeCost,
            limits(),
            &ctx(),
        )
        .unwrap();
        let verified: Vec<_> = report
            .outcomes
            .iter()
            .filter(|o| o.classification == RepairClassification::VerifiedSufficient)
            .collect();
        assert_eq!(verified.len(), 1);
        let robust = SensitivityFinding { robust: true, ..finding };
        let refused =
            SensitivityRepairFamily::try_new(graph, v(0), v(1), "pop", [v(0), v(1), v(2)], robust)
                .unwrap_err();
        assert_eq!(refused.detail, "identification_repair.not_a_failure");
    }
}

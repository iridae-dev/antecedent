//! Compiled continuations of an accepted excitation, and the release question that fits a
//! challenge, offers its kernel to the parameter bridge and draws the release in one program.
//!
//! An intervened continuation adds one event of the accepted kernel's source type (or of a
//! different mark) and draws its offspring together with the observed history's excitation.
//! The paths are a conditional scenario under the kernel the bridge kept; they are not, by
//! themselves, the utility of releasing.
//!
//! When some identified members of the declared structure leave the exposure's edge into the
//! outcome absent or reversed, the law is one weighted mixture: the members that orient the
//! edge draw the continuation with the added event exciting, and the null members draw the same
//! continuation without it, on the same random stream for each path index, so the difference
//! between the two groups is the structure alone. Group weights are the identified members'
//! weights, or their uniform enumeration weight for a form without weights.

use std::sync::Arc;

use antecedent_contract::bridge::BridgeContract;
use antecedent_core::ExecutionContext;
use antecedent_core::execution::{AdmissionLease, RngFactory};
use antecedent_core::identity::SemanticDigest;

use super::challenge::{
    DiagnosticStatus, ExcitationChallenge, ExcitationReading, MarkedExcitationChallenge,
};
use super::discovery::{Mixture, Structural, single_dag, structural_members};
use super::hawkes::{self, INTERVENE};
use super::marked::{self, MarkedHistory};
use super::question::{
    AcceptedExcitation, Continuation, ExcitationSubject, Kernel, ReleaseAction, ReleaseDraw,
    accept, before_release, catalog_of, marked_before, release_window,
};
use super::{
    Arrival, EventError, EventHistory, EventLaw, EventLawMeaning, PreparedKernel, admit, digest,
    history_bytes, path_bytes, validate_budget,
};
use crate::planner::lowered::EventProcessRoute::ParameterBridge;

/// The history an intervened continuation conditions on, owned by its compiled program. A
/// marked history is read as a catalog with one name per prototype.
enum Conditioning {
    Catalog(EventHistory),
    Marked { catalog: EventHistory, marked: MarkedHistory },
}

impl Conditioning {
    fn catalog(&self) -> &EventHistory {
        match self {
            Self::Catalog(history) | Self::Marked { catalog: history, .. } => history,
        }
    }
}

/// The resolved continuation: what it conditions on, the draw with the added event's time, and
/// the added event's type.
struct Resolved {
    conditioning: Conditioning,
    draw: ReleaseDraw,
    kind: usize,
}

/// Check a continuation against the accepted kernel and resolve the release action. `None`
/// adds one event of the accepted source type at `draw.release`.
fn resolve(
    kernel: &Kernel,
    source: usize,
    continuation: Continuation<'_>,
    action: Option<ReleaseAction>,
    draw: ReleaseDraw,
) -> Result<Resolved, EventError> {
    let observed_end = match continuation {
        Continuation::Catalog(history) => history.window.end,
        Continuation::Marked(history) => history.window.end,
    };
    let (release, kind) = match action {
        None => (draw.release, source),
        Some(ReleaseAction::Now) => (observed_end, source),
        Some(ReleaseAction::Delay { release }) => {
            if release.partial_cmp(&observed_end) != Some(std::cmp::Ordering::Greater) {
                return Err(EventError::invalid("a delay releases after the observation"));
            }
            (release, source)
        }
        Some(ReleaseAction::ChangeMark { kind, release }) => {
            if kind == source {
                return Err(EventError::invalid("changing the mark uses a different type"));
            }
            (release, kind)
        }
    };
    let draw = ReleaseDraw { release, ..draw };
    validate_budget(draw.paths, draw.max_events)?;
    release_window(draw)?;
    let conditioning = match (kernel, continuation) {
        (Kernel::Catalog(parameters), Continuation::Catalog(history)) => {
            if history.catalog.len() != parameters.baseline.len() {
                return Err(EventError::invalid(
                    "intervened history catalog does not match the accepted kernel",
                ));
            }
            Conditioning::Catalog(history.clone())
        }
        (Kernel::Marked(parameters, prototypes), Continuation::Marked(history)) => {
            history.validate()?;
            if &history.prototypes != prototypes
                || parameters.decay.len() != history.prototypes[0].source.len()
            {
                return Err(EventError::invalid(
                    "intervened marked history does not match the accepted kernel",
                ));
            }
            Conditioning::Marked { catalog: catalog_of(history), marked: history.clone() }
        }
        _ => {
            return Err(EventError::invalid(
                "an intervened law uses the history kind of the accepted kernel",
            ));
        }
    };
    let catalog = conditioning.catalog();
    catalog.validate(true)?;
    if kind >= catalog.catalog.len() {
        return Err(EventError::invalid("the release mark is outside the catalog"));
    }
    if draw.release <= catalog.empty_origin {
        return Err(EventError::invalid("the added event must follow the empty origin"));
    }
    Ok(Resolved { conditioning, draw, kind })
}

/// Draw the intervened continuations of `resolved` under the dispatch-scoped `ctx`. The law
/// carries the identity of the program that drew it.
///
/// Members that orient the edge draw with the added event; null members draw path `i` on the
/// same stream `i` without it. A law with one group is unweighted, exactly the single-kernel
/// law; a law with both groups weights each path by its group's weight over the path count.
fn sample(
    kernel: &Kernel,
    resolved: &Resolved,
    mixture: Mixture,
    identity: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<EventLaw, EventError> {
    let Resolved { conditioning, draw, kind } = resolved;
    let (draw, kind) = (*draw, *kind);
    let window = release_window(draw)?;
    let catalog = conditioning.catalog();
    let mut work = hawkes::Work::new(ctx);
    let rng = RngFactory::from_seed(draw.seed);
    let groups = mixture.groups();
    let mut paths = Vec::with_capacity(draw.paths * groups.len());
    let mut weights = Vec::with_capacity(draw.paths * groups.len());
    for &(excites, group_weight) in &groups {
        for index in 0..draw.paths {
            let added = Arrival { time: draw.release, kind };
            let offspring = match (kernel, conditioning) {
                (Kernel::Catalog(parameters), _) => {
                    let mut prefix = before_release(catalog, draw.release);
                    if excites {
                        prefix.push(added);
                    }
                    hawkes::sample(
                        catalog,
                        &prefix,
                        parameters,
                        window,
                        (&rng, INTERVENE, index as u64),
                        draw.max_events,
                        &mut work,
                    )?
                }
                (Kernel::Marked(parameters, _), Conditioning::Marked { marked: history, .. }) => {
                    let mut prefix = marked_before(history, draw.release);
                    if excites {
                        prefix.push(marked::MarkedArrival { time: draw.release, prototype: kind });
                    }
                    marked::sample(
                        history,
                        &prefix,
                        parameters,
                        window,
                        (&rng, INTERVENE, index as u64),
                        draw.max_events,
                        &mut work,
                    )?
                    .into_iter()
                    .map(|event| Arrival { time: event.time, kind: event.prototype })
                    .collect()
                }
                (Kernel::Marked(..), Conditioning::Catalog(_)) => {
                    return Err(EventError::invalid(
                        "marked intervention needs the marked history",
                    ));
                }
            };
            let mut path = if excites { vec![added] } else { Vec::new() };
            path.extend(offspring);
            paths.push(path);
            weights.push(group_weight / draw.paths as f64);
        }
    }
    let lease = Arc::new(admit(
        ctx,
        0,
        path_bytes(paths.len(), draw.max_events.saturating_add(1)) + 8_192,
    )?);
    Ok(EventLaw {
        history: Arc::new(catalog.clone()),
        window,
        paths: Arc::new(paths),
        weights: (groups.len() > 1).then(|| Arc::new(weights)),
        path_left: None,
        meaning: EventLawMeaning::Intervened,
        identity,
        provider: None,
        provider_receipt: None,
        model_fit: None,
        refit_parameters: None,
        release_supported: super::release_supported(super::BRIDGE_UNIT),
        _lease: lease,
    })
}

/// Compiled intervened continuation of an accepted excitation, composed on the program whose
/// bridge accepted it.
pub struct PreparedReleaseLaw {
    kernel: Kernel,
    resolved: Resolved,
    mixture: Mixture,
    plan: PreparedKernel,
    _lease: AdmissionLease,
}

fn prepare_draw(
    accepted: &AcceptedExcitation,
    continuation: Continuation<'_>,
    action: Option<ReleaseAction>,
    draw: ReleaseDraw,
    ctx: &ExecutionContext,
) -> Result<PreparedReleaseLaw, EventError> {
    super::super::lowering::refuse_reentry(ctx)?;
    let resolved = resolve(&accepted.kernel, accepted.source, continuation, action, draw)?;
    let lease = admit(ctx, 0, history_bytes(resolved.conditioning.catalog()))?;
    let (catalog, marked) = match &resolved.conditioning {
        Conditioning::Catalog(history) => (Some(history), None),
        Conditioning::Marked { marked, .. } => (None, Some(marked)),
    };
    let draw = resolved.draw;
    let source = digest(
        "antecedent.events.release_law.v1",
        &(
            accepted.identity.to_hex(),
            catalog,
            marked,
            resolved.kind,
            (draw.release, draw.window_end, draw.paths, draw.max_events, draw.seed),
        ),
    )?;
    let plan = PreparedKernel::composed_on(
        ParameterBridge,
        source,
        accepted.identity,
        "release_law",
        None,
    )?;
    let kernel = match &accepted.kernel {
        Kernel::Catalog(parameters) => Kernel::Catalog(parameters.clone()),
        Kernel::Marked(parameters, prototypes) => {
            Kernel::Marked(parameters.clone(), prototypes.clone())
        }
    };
    Ok(PreparedReleaseLaw { kernel, resolved, mixture: accepted.mixture, plan, _lease: lease })
}

impl crate::Study {
    /// Compile one release action of an accepted excitation: one added event now, later, or
    /// under a different mark, drawn with the observed history's excitation.
    ///
    /// # Errors
    /// The action is not a later time or a different mark, the history does not match the
    /// accepted kernel, the draw is outside the event budget, or a resource refusal.
    pub fn prepare_release_law(
        accepted: &AcceptedExcitation,
        continuation: Continuation<'_>,
        action: ReleaseAction,
        draw: ReleaseDraw,
        ctx: &ExecutionContext,
    ) -> Result<PreparedReleaseLaw, EventError> {
        prepare_draw(accepted, continuation, Some(action), draw, ctx)
    }

    /// Compile continuations after one added event of the accepted source type at
    /// `draw.release`.
    ///
    /// # Errors
    /// The history does not match the accepted kernel, the draw is outside the event budget,
    /// or a resource refusal.
    pub fn prepare_intervened_law(
        accepted: &AcceptedExcitation,
        continuation: Continuation<'_>,
        draw: ReleaseDraw,
        ctx: &ExecutionContext,
    ) -> Result<PreparedReleaseLaw, EventError> {
        prepare_draw(accepted, continuation, None, draw, ctx)
    }
}

impl PreparedReleaseLaw {
    /// Inspect the compiled continuation without drawing it.
    pub fn plan_view(&self) -> crate::planner::PlanView {
        self.plan.plan_view()
    }

    /// Draw the intervened continuations.
    ///
    /// # Errors
    /// Resource exhaustion, cancellation, numerical failure, or an unsupported release scope.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<EventLaw, EventError> {
        let (scoped, publication) = super::scoped_execution(ctx, &self.plan)?;
        publication.finish(sample(
            &self.kernel,
            &self.resolved,
            self.mixture,
            self.plan.identity(),
            &scoped,
        ))
    }
}

/// The challenge a release question fits before it offers the constant-baseline kernel to the
/// bridge.
#[derive(Clone, Debug)]
pub enum QuestionChallenge {
    /// A catalog immigrant challenge.
    Catalog(ExcitationChallenge),
    /// A marked immigrant challenge.
    Marked(MarkedExcitationChallenge),
}

/// A release question: fit the challenge, offer its agreed direct mass to the parameter bridge,
/// and draw one release action of the kernel the bridge kept.
#[derive(Clone, Debug)]
pub struct ReleaseQuestion {
    /// The immigrant challenge whose agreed reading the bridge may carry.
    pub challenge: QuestionChallenge,
    /// The parameter-intervention bridge the direct mass is offered to.
    pub bridge: BridgeContract,
    /// The release action drawn when the bridge accepts.
    pub action: ReleaseAction,
    /// How many continuations to draw.
    pub draw: ReleaseDraw,
}

/// What a release question produced: the reading always, and the accepted excitation and its
/// intervened law only when the diagnostic agreed and the bridge accepted.
#[non_exhaustive]
pub struct ReleaseQuestionOutcome {
    /// The immigrant diagnostic of the challenged edge.
    pub reading: ExcitationReading,
    /// The excitation the bridge accepted, when the reading agreed.
    pub accepted: Option<AcceptedExcitation>,
    /// The intervened continuation of the release action, when the bridge accepted.
    pub law: Option<EventLaw>,
}

/// Compiled release question, bound to the declared structure its structural reading cites.
pub struct PreparedReleaseQuestion {
    question: ReleaseQuestion,
    structural: Structural,
    plan: PreparedKernel,
    _lease: AdmissionLease,
}

/// The exposure and outcome names of a challenge: the catalog names, or one name per prototype.
fn challenge_names(challenge: &QuestionChallenge) -> Result<(String, String), EventError> {
    let (names, source, target) = match challenge {
        QuestionChallenge::Catalog(challenge) => {
            (challenge.history.catalog.clone(), challenge.source, challenge.target)
        }
        QuestionChallenge::Marked(challenge) => (
            (0..challenge.history.prototypes.len())
                .map(|index| format!("prototype-{index}"))
                .collect(),
            challenge.source,
            challenge.target,
        ),
    };
    match (names.get(source), names.get(target)) {
        (Some(exposure), Some(outcome)) => Ok((exposure.clone(), outcome.clone())),
        _ => Err(EventError::invalid("the questioned types are outside the catalog")),
    }
}

fn action_wire(action: ReleaseAction) -> (&'static str, Option<usize>, Option<f64>) {
    match action {
        ReleaseAction::Now => ("now", None, None),
        ReleaseAction::Delay { release } => ("delay", None, Some(release)),
        ReleaseAction::ChangeMark { kind, release } => ("change_mark", Some(kind), Some(release)),
    }
}

impl crate::Study {
    /// Compile a release question before the challenge fits run.
    ///
    /// `structure` is any declared structure the shared member carrier takes (a
    /// `&AcceptedGraph` converts); its members are enumerated and identified here.
    ///
    /// # Errors
    /// Invalid challenge, draw outside the event budget, a structure not bound to the catalog
    /// names or none of whose members identifies the reading, or a resource refusal.
    pub fn prepare_release_question(
        question: ReleaseQuestion,
        structure: impl Into<crate::analysis::structure_members::DeclaredStructure>,
        ctx: &ExecutionContext,
    ) -> Result<PreparedReleaseQuestion, EventError> {
        super::super::lowering::refuse_reentry(ctx)?;
        let (catalog, marked, bytes) = match &question.challenge {
            QuestionChallenge::Catalog(challenge) => {
                challenge.validate()?;
                (Some(challenge), None, history_bytes(&challenge.history))
            }
            QuestionChallenge::Marked(challenge) => {
                challenge.validate()?;
                (None, Some(challenge), 8_192 + challenge.history.events.len() as u64 * 64)
            }
        };
        validate_budget(question.draw.paths, question.draw.max_events)?;
        let lease = admit(ctx, 0, bytes)?;
        let structure = structure.into();
        let (exposure, outcome) = challenge_names(&question.challenge)?;
        let structural = structural_members(&structure, &exposure, &outcome, ctx)?;
        let draw = question.draw;
        let provenance = super::discovery::provenance_hex(&structure);
        let source = if single_dag(&structure) {
            digest(
                "antecedent.events.release_question.v1",
                &(
                    catalog,
                    marked,
                    question.bridge.identity().to_hex(),
                    action_wire(question.action),
                    (draw.release, draw.window_end, draw.paths, draw.max_events, draw.seed),
                    provenance,
                ),
            )?
        } else {
            digest(
                "antecedent.events.release_question.v2",
                &(
                    catalog,
                    marked,
                    question.bridge.identity().to_hex(),
                    action_wire(question.action),
                    (draw.release, draw.window_end, draw.paths, draw.max_events, draw.seed),
                    provenance,
                    structural.members.identity().to_hex(),
                    super::discovery::ReadingMode::declared(&question.bridge).functional(),
                    super::discovery::CYCLIC.as_str(),
                ),
            )?
        };
        let plan = PreparedKernel::on(ParameterBridge, source, "release_question")?;
        Ok(PreparedReleaseQuestion { question, structural, plan, _lease: lease })
    }
}

impl PreparedReleaseQuestion {
    /// Inspect the compiled question without fitting.
    pub fn plan_view(&self) -> crate::planner::PlanView {
        self.plan.plan_view()
    }

    /// The evidence the structural reading cites, as lowercase hex: the provenance digest of
    /// the search the structure came from, or the declared structure's canonical identity.
    #[must_use]
    pub fn evidence(&self) -> Option<String> {
        Some(self.structural.evidence.clone())
    }

    /// The members of the declared structure and their identification of the reading.
    #[must_use]
    pub fn members(&self) -> &crate::analysis::structure_members::StructureMembers {
        &self.structural.members
    }

    /// Fit the challenge, offer an agreed direct mass to the bridge and draw the release.
    ///
    /// A withheld reading returns the reading alone: no claim and no intervened law.
    ///
    /// # Errors
    /// A fit fails, the bridge or the discovery refuses an agreed mass, the release action is
    /// not a later time or a different mark, or an unsupported release scope.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<ReleaseQuestionOutcome, EventError> {
        let (scoped, publication) = super::scoped_execution(ctx, &self.plan)?;
        let (reading, subject) = match &self.question.challenge {
            QuestionChallenge::Catalog(challenge) => {
                let (reading, constant) = challenge.run(&scoped)?;
                (reading, challenge.subject(constant))
            }
            QuestionChallenge::Marked(challenge) => {
                let (reading, constant) = challenge.run(&scoped)?;
                (reading, challenge.subject(constant))
            }
        };
        if reading.status != DiagnosticStatus::Agreed {
            return Ok(ReleaseQuestionOutcome { reading, accepted: None, law: None });
        }
        let identity = self.plan.identity();
        let accepted =
            accept(&reading, &subject, &self.question.bridge, &self.structural, identity, &scoped)?;
        let continuation = match &subject {
            ExcitationSubject::Catalog { history, .. } => Continuation::Catalog(history),
            ExcitationSubject::Marked { history, .. } => Continuation::Marked(history),
        };
        let resolved = resolve(
            &accepted.kernel,
            accepted.source,
            continuation,
            Some(self.question.action),
            self.question.draw,
        )?;
        let law = sample(&accepted.kernel, &resolved, accepted.mixture, identity, &scoped)?;
        publication.finish(Ok(ReleaseQuestionOutcome {
            reading,
            accepted: Some(accepted),
            law: Some(law),
        }))
    }
}

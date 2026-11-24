//! Submit a catalog or marked excitation mass to the parameter-intervention bridge.
//!
//! The immigrant diagnostic is a robustness check. This module does not treat
//! agreement as identification. An agreed direct mass can cross
//! [`BridgeKind::ParameterInterventionResponse`] when an accepted discovery
//! orients the exposure into the outcome. That provenance is the evidence.
//! Descendants stay a
//! conditional intensity: the resolvent is nonlinear, so the affine bridge
//! does not carry them as a mean. A withheld diagnostic, a missing reading, or
//! a bridge that asks for anything else refuses, and no intervened law is
//! produced for the decision.

use std::sync::Arc;

use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeScope, BridgeSource, QuantityClass, ValueKind, derive_bridged_claim,
};
use antecedent_contract::{
    ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus, QuantityRole, ReasoningView,
    ScientificQuantity, SlotAvailability,
};
use antecedent_core::ExecutionContext;
use antecedent_core::execution::{AdmissionLease, RngFactory};
use antecedent_core::identity::{ContractIdentities, SemanticDigest};
use serde::Serialize;

use super::challenge::{DiagnosticStatus, ExcitationReading};
use super::discovery::structural_evidence;
use super::hawkes::{self, HawkesParameters, INTERVENE};
use super::marked::{self, MarkPrototype, MarkedHistory, MarkedParameters};
use super::{
    Arrival, EventError, EventHistory, EventLaw, EventLawMeaning, EventWindow, PreparedKernel,
    admit, digest, path_bytes, poll, validate_budget,
};

/// The kernel whose agreed direct mass is offered to the bridge.
#[derive(Clone, Debug)]
pub enum ExcitationSubject {
    /// Finite catalog, the history it was fit on, and the column cap of that fit.
    Catalog {
        /// Fitted coefficients.
        parameters: HawkesParameters,
        /// History the Gram and the cap are checked against.
        history: EventHistory,
        /// Source-column cap of the fit, strictly inside `(0, 1)`.
        column_cap: f64,
    },
    /// Finite-rank prototypes.
    Marked {
        /// Fitted coefficients.
        parameters: MarkedParameters,
        /// Prototypes the coefficients were fitted against.
        prototypes: Vec<MarkPrototype>,
        /// History the Gram and the reproduction cap are checked against.
        history: MarkedHistory,
        /// Reproduction cap of the fit, strictly inside `(0, 1)`.
        reproduction_cap: f64,
    },
}

/// Which intervened immigrant the release decision draws.
#[derive(Clone, Copy, Debug)]
pub enum ReleaseAction {
    /// One event of the questioned type at the end of the observation.
    Now,
    /// One event of the questioned type after the observation has ended.
    Delay {
        /// Time of the added event.
        release: f64,
    },
    /// One event of a different type or prototype.
    ChangeMark {
        /// Catalog index or prototype of the added event.
        kind: usize,
        /// Time of the added event.
        release: f64,
    },
}

/// History the intervened continuation conditions on.
pub enum Continuation<'a> {
    /// Catalog observation.
    Catalog(&'a EventHistory),
    /// Marked observation. The law's catalog is one name per prototype.
    Marked(&'a MarkedHistory),
}

/// How many intervened continuations to draw, and where the added event sits.
#[derive(Clone, Copy, Debug)]
pub struct ReleaseDraw {
    /// Time of the one added event.
    pub release: f64,
    /// End of the continuation window.
    pub window_end: f64,
    /// Independent continuations.
    pub paths: usize,
    /// Maximum offspring per continuation, not counting the added event.
    pub max_events: usize,
    /// Stream seed.
    pub seed: u64,
}

struct Mass {
    direct: f64,
    horizon_descendants: f64,
    lifetime_descendants: f64,
}

enum Kernel {
    Catalog(HawkesParameters),
    Marked(MarkedParameters, Vec<MarkPrototype>),
}

/// A question whose direct mass the parameter bridge accepted.
///
/// `bridged_direct` is the bridge's mean. The descendant fields are functionals
/// of the conditional intensity and are not that mean.
pub struct AcceptedExcitation {
    /// Mapped direct mass.
    pub bridged_direct: f64,
    /// Constant-baseline horizon direct mass that the bridge carried.
    pub conditional_direct: f64,
    /// Filter-ODE descendants on the challenged horizon. Not a bridged mean.
    pub horizon_descendants: f64,
    /// Lifetime resolvent descendants, excluding the initiator. Not a bridged mean.
    pub lifetime_descendants: f64,
    /// The bridge carried evidence established in another scope, under a premise.
    pub transported: bool,
    kernel: Kernel,
    source: usize,
    identity: SemanticDigest,
    _lease: AdmissionLease,
}

/// Compiled question, checked before the bridge derives a claim.
pub struct PreparedExcitationQuestion {
    reading: ExcitationReading,
    subject: ExcitationSubject,
    bridge: BridgeContract,
    discovery: crate::AcceptedGraph,
    plan: PreparedKernel,
    _lease: AdmissionLease,
}

/// A quantity named on the parameter-intervention bridge, without a class or a role.
#[derive(Clone, Debug)]
pub struct DeclaredQuantity {
    /// Stable coordinate of the quantity.
    pub variable_id: String,
    /// Name shown with the claim.
    pub variable_name: String,
    /// Unit of the direct mass.
    pub units: String,
    /// Population the quantity was read in.
    pub population_id: String,
    /// Regime of this endpoint.
    pub regime_id: String,
    /// Horizon the functional is taken over.
    pub horizon: u32,
    /// Must be `direct_offspring` for this bridge.
    pub functional_id: String,
}

/// Build the parameter-intervention contract the excitation question can offer.
///
/// Evidence is left empty. The accepted discovery writes its provenance digest
/// in before the bridge is asked to carry the mean.
///
/// # Errors
/// The map is neither an identity nor a complete affine map.
pub fn parameter_bridge(
    source: DeclaredQuantity,
    target: DeclaredQuantity,
    intercept: Option<f64>,
    slope: Option<f64>,
    statement: &str,
) -> Result<BridgeContract, EventError> {
    let mapping = match (intercept, slope) {
        (None, None) => BridgeMapping::Identity,
        (Some(intercept), Some(slope)) => BridgeMapping::Affine { intercept, slope },
        _ => {
            return Err(EventError::Fit("an affine map needs both an intercept and a slope"));
        }
    };
    let source_quantity = declared_quantity(source);
    let target_quantity = declared_quantity(target);
    Ok(BridgeContract {
        kind: BridgeKind::ParameterInterventionResponse,
        source: BridgeEndpoint {
            quantity: source_quantity.clone(),
            class: QuantityClass::ModelParameter,
        },
        target: BridgeEndpoint {
            quantity: target_quantity,
            class: QuantityClass::InterventionResponse,
        },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope {
            population_id: source_quantity.population_id.clone(),
            regime_id: source_quantity.regime_id.clone(),
        },
        mapping,
        assumptions: vec![BridgeAssumption {
            id: "structural_kernel".into(),
            kind: AssumptionKind::ParameterInterpretation,
            statement: statement.to_owned(),
            evidence: None,
        }],
        transport: None,
        mapping_uncertainty: None,
        source_dependence: None,
        estimand_bridge: None,
    })
}

fn declared_quantity(quantity: DeclaredQuantity) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: quantity.variable_id,
        variable_name: quantity.variable_name,
        role: QuantityRole::Outcome,
        units: quantity.units,
        population_id: quantity.population_id,
        regime_id: quantity.regime_id,
        horizon: quantity.horizon,
        functional_id: quantity.functional_id,
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

impl crate::Study {
    /// Compile an excitation question for the parameter-intervention bridge.
    ///
    /// # Errors
    /// The diagnostic was withheld, the kernel does not reproduce it, the
    /// contract asks the bridge to carry a nonlinear descendant, the discovery
    /// does not orient the edge, or the bridge refuses the declaration.
    pub fn prepare_excitation_question(
        reading: ExcitationReading,
        subject: ExcitationSubject,
        bridge: BridgeContract,
        discovery: &crate::AcceptedGraph,
        ctx: &ExecutionContext,
    ) -> Result<PreparedExcitationQuestion, EventError> {
        super::super::lowering::refuse_reentry(ctx)?;
        let lease = admit(ctx, 0, 8_192)?;
        let source = digest(
            "antecedent.events.excitation_question.v1",
            &(
                &reading,
                bridge.kind.as_str(),
                bridge.source.quantity.canonical_identity(),
                bridge.target.quantity.canonical_identity(),
                discovery.provenance().map(crate::accepted::StructureProvenance::digest),
            ),
        )?;
        let plan = PreparedKernel::new(source, "excitation_question", super::CAPABILITY)?;
        Ok(PreparedExcitationQuestion {
            reading,
            subject,
            bridge,
            discovery: discovery.clone(),
            plan,
            _lease: lease,
        })
    }
}

impl PreparedExcitationQuestion {
    /// Inspect the compiled question.
    pub fn plan_view(&self) -> crate::planner::PlanView {
        self.plan.plan_view()
    }

    /// Ask the parameter bridge to carry the agreed direct mass.
    ///
    /// # Errors
    /// The diagnostic, the kernel, or the bridge contract does not support
    /// the question. A refusal produces no claim and no intervened law.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<AcceptedExcitation, EventError> {
        let _scoped = super::scoped_execution(ctx, &self.plan)?;
        let mass = offered_mass(&self.reading, &self.subject)?;
        kernel_gate(&self.subject, &self.reading, ctx)?;
        let (exposure, outcome) = pair_names(&self.subject, &self.reading)?;
        let evidence = structural_evidence(&self.discovery, &exposure, &outcome)?;
        let bridge = cite_discovery(&self.bridge, &evidence)?;
        direct_bridge(&bridge)?;
        let identities = ContractIdentities::new(
            self.plan.identity(),
            self.plan.identity(),
            None,
            None,
            self.plan.identity(),
            self.plan.identity(),
            self.plan.identity(),
        );
        let claim = point_claim(&identities, &bridge.source.quantity, mass.direct);
        let derived = derive_bridged_claim(
            &bridge,
            &BridgeSource { claim: &claim, quantity: &bridge.source.quantity },
        )
        .map_err(|refusal| {
            EventError::invalid(format!(
                "{}: {}",
                refusal.detail,
                refusal.remedy.unwrap_or("the bridge names the declaration that would support it")
            ))
        })?;
        Ok(AcceptedExcitation {
            bridged_direct: derived.value,
            conditional_direct: mass.direct,
            horizon_descendants: mass.horizon_descendants,
            lifetime_descendants: mass.lifetime_descendants,
            transported: derived.transported,
            kernel: kernel_of(&self.subject),
            source: self.reading.source,
            identity: self.plan.identity(),
            _lease: admit(ctx, 0, 8_192)?,
        })
    }
}

impl AcceptedExcitation {
    /// Catalog coefficients the bridge accepted, when the kernel is a catalog.
    #[must_use]
    pub fn catalog_parameters(&self) -> Option<&HawkesParameters> {
        match &self.kernel {
            Kernel::Catalog(parameters) => Some(parameters),
            Kernel::Marked(_, _) => None,
        }
    }

    /// Draw continuations after one added source event.
    ///
    /// Pass the catalog history for a catalog kernel and the marked history for
    /// a marked kernel. The law's catalog is that history, or one name per
    /// prototype. It can drive the existing paired worlds. The paths exist
    /// only because this question was accepted.
    ///
    /// # Errors
    /// The history does not match the accepted kernel, or the draw is outside
    /// the event budget.
    pub fn intervened_law(
        &self,
        continuation: Continuation<'_>,
        draw: ReleaseDraw,
        ctx: &ExecutionContext,
    ) -> Result<EventLaw, EventError> {
        self.draw_intervened(continuation, draw, self.source, ctx)
    }

    /// Draw one release action: now, later, or under a different mark.
    ///
    /// Each action is one added event. The excitation coefficients stay the
    /// accepted kernel.
    ///
    /// # Errors
    /// The action is not a later time or a different mark, or the draw is
    /// outside the event budget.
    pub fn release_law(
        &self,
        continuation: Continuation<'_>,
        action: ReleaseAction,
        draw: ReleaseDraw,
        ctx: &ExecutionContext,
    ) -> Result<EventLaw, EventError> {
        let observed_end = match continuation {
            Continuation::Catalog(history) => history.window.end,
            Continuation::Marked(history) => history.window.end,
        };
        let (release, kind) = match action {
            ReleaseAction::Now => (observed_end, self.source),
            ReleaseAction::Delay { release } => {
                if release.partial_cmp(&observed_end) != Some(std::cmp::Ordering::Greater) {
                    return Err(EventError::invalid("a delay releases after the observation"));
                }
                (release, self.source)
            }
            ReleaseAction::ChangeMark { kind, release } => {
                if kind == self.source {
                    return Err(EventError::invalid("changing the mark uses a different type"));
                }
                (release, kind)
            }
        };
        self.draw_intervened(continuation, ReleaseDraw { release, ..draw }, kind, ctx)
    }

    fn draw_intervened(
        &self,
        continuation: Continuation<'_>,
        draw: ReleaseDraw,
        kind: usize,
        ctx: &ExecutionContext,
    ) -> Result<EventLaw, EventError> {
        poll(ctx)?;
        validate_budget(draw.paths, draw.max_events)?;
        let window = release_window(draw)?;
        let (catalog, marked_history) = match (&self.kernel, continuation) {
            (Kernel::Catalog(parameters), Continuation::Catalog(history)) => {
                if history.catalog.len() != parameters.baseline.len() {
                    return Err(EventError::invalid(
                        "intervened history catalog does not match the accepted kernel",
                    ));
                }
                (history.clone(), None)
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
                (catalog_of(history), Some(history))
            }
            _ => {
                return Err(EventError::invalid(
                    "an intervened law uses the history kind of the accepted kernel",
                ));
            }
        };
        catalog.validate(true)?;
        if kind >= catalog.catalog.len() {
            return Err(EventError::invalid("the release mark is outside the catalog"));
        }
        if draw.release <= catalog.empty_origin {
            return Err(EventError::invalid("the added event must follow the empty origin"));
        }
        let mut work = hawkes::Work::new(ctx);
        let rng = RngFactory::from_seed(draw.seed);
        let mut paths = Vec::with_capacity(draw.paths);
        for index in 0..draw.paths {
            let mut path = vec![Arrival { time: draw.release, kind }];
            let offspring = match &self.kernel {
                Kernel::Catalog(parameters) => {
                    let mut prefix = before_release(&catalog, draw.release);
                    prefix.push(path[0]);
                    hawkes::sample(
                        &catalog,
                        &prefix,
                        parameters,
                        window,
                        (&rng, INTERVENE, index as u64),
                        draw.max_events,
                        &mut work,
                    )?
                }
                Kernel::Marked(parameters, _) => {
                    let marked_history = marked_history.ok_or_else(|| {
                        EventError::invalid("marked intervention needs the marked history")
                    })?;
                    let mut prefix = marked_before(marked_history, draw.release);
                    prefix.push(marked::MarkedArrival { time: draw.release, prototype: kind });
                    marked::sample(
                        marked_history,
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
            };
            path.extend(offspring);
            paths.push(path);
        }
        let lease = Arc::new(admit(
            ctx,
            0,
            path_bytes(draw.paths, draw.max_events.saturating_add(1)) + 8_192,
        )?);
        Ok(EventLaw {
            history: Arc::new(catalog),
            window,
            paths: Arc::new(paths),
            weights: None,
            path_left: None,
            meaning: EventLawMeaning::Intervened,
            identity: self.identity,
            provider: None,
            provider_receipt: None,
            model_fit: None,
            refit_parameters: None,
            release_supported: super::release_supported(super::CAPABILITY),
            _lease: lease,
        })
    }
}

fn offered_mass(
    reading: &ExcitationReading,
    subject: &ExcitationSubject,
) -> Result<Mass, EventError> {
    if reading.status != DiagnosticStatus::Agreed {
        return Err(EventError::Fit(
            "immigrant diagnostic did not agree; the mass stays a conditional intensity",
        ));
    }
    let mass = match subject {
        ExcitationSubject::Catalog { parameters, .. } => {
            let offspring = parameters.excitation_offspring(reading.source, reading.horizon)?;
            Mass {
                direct: component(&offspring.direct, reading.target)?,
                horizon_descendants: component(&offspring.horizon_descendants, reading.target)?,
                lifetime_descendants: component(&offspring.lifetime_descendants, reading.target)?,
            }
        }
        ExcitationSubject::Marked { parameters, prototypes, .. } => {
            let offspring =
                parameters.excitation_offspring(prototypes, reading.source, reading.horizon)?;
            Mass {
                direct: component(&offspring.direct, reading.target)?,
                horizon_descendants: component(&offspring.horizon_descendants, reading.target)?,
                lifetime_descendants: component(&offspring.lifetime_descendants, reading.target)?,
            }
        }
    };
    if (mass.direct - reading.conditional_direct).abs() > 1e-8 {
        return Err(EventError::invalid(
            "question kernel does not reproduce the agreed direct mass",
        ));
    }
    Ok(mass)
}

fn component(values: &[f64], target: usize) -> Result<f64, EventError> {
    values
        .get(target)
        .copied()
        .ok_or_else(|| EventError::invalid("the questioned target is outside the kernel"))
}

fn direct_bridge(bridge: &BridgeContract) -> Result<(), EventError> {
    if bridge.kind != BridgeKind::ParameterInterventionResponse
        || bridge.source.class != QuantityClass::ModelParameter
        || bridge.target.class != QuantityClass::InterventionResponse
        || bridge.source_value != ValueKind::Mean
        || bridge.target_value != ValueKind::Mean
    {
        return Err(EventError::Fit(
            "an excitation mass enters the decision loop only as a parameter-intervention mean",
        ));
    }
    if bridge.source.quantity.functional_id != "direct_offspring"
        || bridge.target.quantity.functional_id != "direct_offspring"
    {
        return Err(EventError::Fit(
            "only the horizon direct mass is affine; descendants stay a conditional intensity",
        ));
    }
    if !matches!(bridge.mapping, BridgeMapping::Identity | BridgeMapping::Affine { .. }) {
        return Err(EventError::Fit(
            "the parameter bridge carries the direct mass by an identity or affine map",
        ));
    }
    match bridge
        .assumptions
        .iter()
        .find(|item| item.kind == AssumptionKind::ParameterInterpretation)
    {
        Some(reading) if reading.evidence.as_ref().is_some_and(|text| !text.trim().is_empty()) => {}
        _ => {
            return Err(EventError::Fit(
                "a structural reading of the excitation mass must name its evidence",
            ));
        }
    }
    if bridge.source.quantity.regime_id != "observational"
        || bridge.source.quantity.regime_id == bridge.target.quantity.regime_id
    {
        return Err(EventError::Fit(
            "the fitted mass is observational and the response must name a different regime",
        ));
    }
    Ok(())
}

fn kernel_of(subject: &ExcitationSubject) -> Kernel {
    match subject {
        ExcitationSubject::Catalog { parameters, .. } => Kernel::Catalog(parameters.clone()),
        ExcitationSubject::Marked { parameters, prototypes, .. } => {
            Kernel::Marked(parameters.clone(), prototypes.clone())
        }
    }
}

fn cite_discovery(bridge: &BridgeContract, evidence: &str) -> Result<BridgeContract, EventError> {
    let mut bridge = bridge.clone();
    let Some(reading) = bridge
        .assumptions
        .iter_mut()
        .find(|item| item.kind == AssumptionKind::ParameterInterpretation)
    else {
        return Err(EventError::Fit(
            "a structural reading of the excitation mass must name its evidence",
        ));
    };
    if reading.evidence.as_ref().is_some_and(|text| text != evidence) {
        return Err(EventError::Fit(
            "the structural evidence is the accepted discovery, not a separate label",
        ));
    }
    reading.evidence = Some(evidence.to_owned());
    Ok(bridge)
}

fn pair_names(
    subject: &ExcitationSubject,
    reading: &ExcitationReading,
) -> Result<(String, String), EventError> {
    let names = match subject {
        ExcitationSubject::Catalog { history, .. } => history.catalog.clone(),
        ExcitationSubject::Marked { history, .. } => {
            (0..history.prototypes.len()).map(|index| format!("prototype-{index}")).collect()
        }
    };
    let exposure = names
        .get(reading.source)
        .cloned()
        .ok_or_else(|| EventError::invalid("the questioned source is outside the kernel"))?;
    let outcome = names
        .get(reading.target)
        .cloned()
        .ok_or_else(|| EventError::invalid("the questioned target is outside the kernel"))?;
    Ok((exposure, outcome))
}

fn kernel_gate(
    subject: &ExcitationSubject,
    reading: &ExcitationReading,
    ctx: &ExecutionContext,
) -> Result<(), EventError> {
    let mut work = hawkes::Work::new(ctx);
    match subject {
        ExcitationSubject::Catalog { parameters, history, column_cap } => {
            if !(0.0..1.0).contains(column_cap) {
                return Err(EventError::invalid("the column cap stays inside (0, 1)"));
            }
            if hawkes::singular_gram(history, &parameters.decay, &mut work)? {
                return Err(EventError::Fit("the excitation Gram is singular"));
            }
            let n = parameters.baseline.len();
            let column: f64 =
                (0..n).map(|target| parameters.branching[target * n + reading.source]).sum();
            if column >= column_cap - 1e-7 {
                return Err(EventError::Fit("the source column is on its stability cap"));
            }
            if reading.source != reading.target {
                direction_gate(
                    edge(parameters, reading.source, reading.target),
                    edge(parameters, reading.target, reading.source),
                )?;
            }
            Ok(())
        }
        ExcitationSubject::Marked { parameters, prototypes, history, reproduction_cap } => {
            if !(0.0..1.0).contains(reproduction_cap) {
                return Err(EventError::invalid("the reproduction cap stays inside (0, 1)"));
            }
            if marked::singular_gram(history, &parameters.decay, &mut work)? {
                return Err(EventError::Fit("the excitation Gram is singular"));
            }
            let forward = lifetime_direct(parameters, prototypes, reading.source, reading.target)?;
            let reverse = lifetime_direct(parameters, prototypes, reading.target, reading.source)?;
            let offspring = parameters.excitation_offspring(prototypes, reading.source, 1.0)?;
            let column: f64 = offspring.lifetime_direct.iter().sum();
            if column >= reproduction_cap - 1e-7 {
                return Err(EventError::Fit("the source prototype is on its reproduction cap"));
            }
            if reading.source != reading.target {
                direction_gate(forward, reverse)?;
            }
            Ok(())
        }
    }
}

fn edge(parameters: &HawkesParameters, source: usize, target: usize) -> f64 {
    let n = parameters.baseline.len();
    parameters.branching[target * n + source]
}

fn lifetime_direct(
    parameters: &MarkedParameters,
    prototypes: &[MarkPrototype],
    source: usize,
    target: usize,
) -> Result<f64, EventError> {
    let offspring = parameters.excitation_offspring(prototypes, source, 1.0)?;
    component(&offspring.lifetime_direct, target)
}

fn direction_gate(forward: f64, reverse: f64) -> Result<(), EventError> {
    if reverse > forward + 1e-8 {
        return Err(EventError::Fit("the queried edge is a lead of the other type"));
    }
    if forward > 1e-8 && reverse > 1e-8 && (forward - reverse).abs() <= 1e-8 {
        return Err(EventError::Fit("the kernel is symmetric on the queried edge"));
    }
    Ok(())
}

fn point_claim(
    identities: &ContractIdentities,
    quantity: &ScientificQuantity,
    value: f64,
) -> ClaimEnvelope {
    ClaimEnvelope::new(
        SemanticDigest::from_bytes([0; 32]),
        *identities,
        ClaimKind::Point,
        Some(value),
        Some(quantity.units.as_str().into()),
        ReasoningView::new(
            SlotAvailability::unavailable("not_identified"),
            SlotAvailability::unavailable("not_identified"),
            SlotAvailability::unavailable("not_identified"),
            SlotAvailability::unavailable("not_identified"),
        ),
        ClaimDomains::new(DomainStatus::NotClaimed, DomainStatus::Unknown, DomainStatus::Evaluated),
        None,
        Arc::<[Arc<str>]>::from([]),
    )
}

fn release_window(draw: ReleaseDraw) -> Result<EventWindow, EventError> {
    let window = EventWindow { start: draw.release, end: draw.window_end };
    window.validate()?;
    Ok(window)
}

fn before_release(history: &EventHistory, release: f64) -> Vec<Arrival> {
    history
        .left
        .iter()
        .chain(&history.events)
        .copied()
        .filter(|event| event.time < release)
        .collect()
}

fn marked_before(history: &MarkedHistory, release: f64) -> Vec<marked::MarkedArrival> {
    history
        .left
        .iter()
        .chain(&history.events)
        .copied()
        .filter(|event| event.time < release)
        .collect()
}

fn catalog_of(history: &MarkedHistory) -> EventHistory {
    EventHistory {
        catalog: (0..history.prototypes.len()).map(|index| format!("prototype-{index}")).collect(),
        time_unit: history.time_unit.clone(),
        origin: history.origin.clone(),
        source: history.source.clone(),
        empty_origin: history.empty_origin,
        window: history.window,
        left: history
            .left
            .iter()
            .map(|event| Arrival { time: event.time, kind: event.prototype })
            .collect(),
        events: history
            .events
            .iter()
            .map(|event| Arrival { time: event.time, kind: event.prototype })
            .collect(),
        gaps: Vec::new(),
    }
}

/// Counts inside the lag the excitation proposed.
///
/// The shock moves the exposure and is excluded from the outcome. The
/// excitation coefficient is not this estimand; an ordinary identified query
/// on these counts is.
#[derive(Clone, Debug, Serialize)]
pub struct LagCountQuery {
    /// Exposure variable, the type the shock moves.
    pub exposure: String,
    /// Outcome variable, the type the shock is excluded from.
    pub outcome: String,
    /// Lag the excitation coefficient proposed.
    pub horizon: f64,
    /// Name of the excluded shock.
    pub instrument: String,
}

/// A shock declared on one type and kept out of another.
#[derive(Clone, Debug)]
pub struct ExcludedShock {
    /// Complete catalog history.
    pub history: EventHistory,
    /// Nonnegative feature that varies across the window.
    pub feature: hawkes::BaselineFeature,
    /// Type the shock moves.
    pub source: usize,
    /// Type the shock is excluded from.
    pub target: usize,
    /// Lag window proposed by the excitation.
    pub horizon: f64,
    /// Stable name of the shock.
    pub instrument: String,
}

/// Compiled handoff from a Hawkes coefficient to a count query.
pub struct PreparedShock {
    shock: ExcludedShock,
    plan: PreparedKernel,
    _lease: AdmissionLease,
}

impl crate::Study {
    /// Compile a shock that leaves the Hawkes parameter.
    ///
    /// # Errors
    /// The feature does not vary, the types coincide, or the horizon is not positive.
    pub fn prepare_shock_query(
        shock: ExcludedShock,
        ctx: &ExecutionContext,
    ) -> Result<PreparedShock, EventError> {
        super::super::lowering::refuse_reentry(ctx)?;
        shock.history.validate(true)?;
        shock.feature.validate(shock.history.window)?;
        if shock.source == shock.target
            || shock.source >= shock.history.catalog.len()
            || shock.target >= shock.history.catalog.len()
            || !shock.horizon.is_finite()
            || shock.horizon <= 0.0
            || shock.instrument.trim().is_empty()
        {
            return Err(EventError::invalid(
                "a shock moves one type, stays out of another, and names a positive lag",
            ));
        }
        let lease = admit(ctx, 0, 8_192)?;
        let source = digest("antecedent.events.shock_query.v1", &shock.instrument)?;
        let plan = PreparedKernel::new(source, "shock_query", super::CAPABILITY)?;
        Ok(PreparedShock { shock, plan, _lease: lease })
    }
}

impl PreparedShock {
    /// Inspect the compiled handoff.
    pub fn plan_view(&self) -> crate::planner::PlanView {
        self.plan.plan_view()
    }

    /// The count query the coefficient proposed. It does not return an excitation mass.
    ///
    /// # Errors
    /// Execution was cancelled or the route is unsupported.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<LagCountQuery, EventError> {
        let _scoped = super::scoped_execution(ctx, &self.plan)?;
        Ok(LagCountQuery {
            exposure: self.shock.history.catalog[self.shock.source].clone(),
            outcome: self.shock.history.catalog[self.shock.target].clone(),
            horizon: self.shock.horizon,
            instrument: self.shock.instrument.clone(),
        })
    }
}

#[cfg(test)]
mod question_tests {
    use super::{Continuation, ExcitationSubject, ReleaseAction, ReleaseDraw};
    use crate::AcceptedGraph;
    use crate::Study;
    use crate::accepted::StructureProvenance;
    use crate::analysis::events::challenge::{DiagnosticStatus, ExcitationReading};
    use crate::analysis::events::hawkes::{BaselineFeature, HawkesParameters};
    use crate::analysis::events::marked::{
        MarkPrototype, MarkedArrival, MarkedHistory, MarkedParameters,
    };
    use crate::analysis::events::{EventHistory, EventLawMeaning, EventWindow};
    use antecedent_contract::bridge::{
        AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind,
        BridgeMapping, BridgeScope, QuantityClass, ValueKind,
    };
    use antecedent_contract::{QuantityRole, ScientificQuantity};
    use antecedent_core::CausalSchemaBuilder;
    use antecedent_core::ExecutionContext;
    use antecedent_graph::Dag;
    use std::sync::Arc;

    fn quantity(regime: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "edge.release_to_activity".into(),
            variable_name: "direct offspring".into(),
            role: QuantityRole::Outcome,
            units: "count".into(),
            population_id: "observed_stream".into(),
            regime_id: regime.into(),
            horizon: 1,
            functional_id: "direct_offspring".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        }
    }

    fn bridge(functional: &str, evidence: Option<&str>, target_regime: &str) -> BridgeContract {
        let mut source = quantity("observational");
        let mut target = quantity(target_regime);
        source.functional_id = functional.into();
        target.functional_id = functional.into();
        BridgeContract {
            kind: BridgeKind::ParameterInterventionResponse,
            source: BridgeEndpoint {
                quantity: source.clone(),
                class: QuantityClass::ModelParameter,
            },
            target: BridgeEndpoint { quantity: target, class: QuantityClass::InterventionResponse },
            source_value: ValueKind::Mean,
            target_value: ValueKind::Mean,
            evidence_scope: BridgeScope {
                population_id: source.population_id.clone(),
                regime_id: source.regime_id.clone(),
            },
            mapping: BridgeMapping::Affine { intercept: 0.0, slope: 2.0 },
            assumptions: vec![BridgeAssumption {
                id: "structural_kernel".into(),
                kind: AssumptionKind::ParameterInterpretation,
                statement: "one added event yields this direct mass under the declared kernel"
                    .into(),
                evidence: evidence.map(str::to_owned),
            }],
            transport: None,
            mapping_uncertainty: None,
            source_dependence: None,
            estimand_bridge: None,
        }
    }

    fn agreed(alpha: f64) -> ExcitationReading {
        let direct = alpha * (1.0 - (-1.0_f64).exp());
        ExcitationReading {
            status: DiagnosticStatus::Agreed,
            reason: None,
            source: 0,
            target: 1,
            horizon: 1.0,
            conditional_direct: direct,
            alternative_direct: direct,
            lifetime_direct: alpha,
            lifetime_descendants: alpha / (1.0 - alpha),
            horizon_descendants: 0.0,
        }
    }

    fn history() -> EventHistory {
        EventHistory {
            catalog: vec!["release".into(), "activity".into()],
            time_unit: "hour".into(),
            origin: "origin".into(),
            source: "log".into(),
            empty_origin: 0.0,
            window: EventWindow { start: 0.0, end: 1.0 },
            left: vec![],
            events: (1..9)
                .map(|index| crate::analysis::events::Arrival {
                    time: 0.1 * f64::from(index),
                    kind: usize::try_from(index).unwrap_or(0) % 2,
                })
                .collect(),
            gaps: vec![],
        }
    }

    fn subject(alpha: f64) -> ExcitationSubject {
        ExcitationSubject::Catalog {
            parameters: HawkesParameters {
                baseline: vec![0.2, 0.2],
                branching: vec![0.0, 0.0, alpha, 0.0],
                decay: vec![0.4, 1.7, 1.0, 0.3],
            },
            history: history(),
            column_cap: 0.9,
        }
    }

    fn discovered(from: &str, to: &str) -> AcceptedGraph {
        let schema = CausalSchemaBuilder::new()
            .continuous(from)
            .treatment()
            .continuous(to)
            .outcome()
            .build()
            .unwrap();
        let dag = Dag::from_named_edges(&schema, &[(from, to)]).unwrap();
        AcceptedGraph::dag(dag).with_schema(&schema).with_provenance(Arc::new(
            StructureProvenance {
                method: "pc".into(),
                inference: "frequentist".into(),
                discovered_class: "dag".into(),
                query: [1; 32],
                search_bounds: "alpha=0.05;max_cond_size=1;max_lag=1;schedule=none".into(),
                data_snapshot: [2; 32],
                program: [3; 32],
                claim: [4; 32],
                review: "accepted".into(),
                resolved_marks: 0,
                structure_prior: None,
                pooling: None,
            },
        ))
    }

    fn ask(
        reading: ExcitationReading,
        alpha: f64,
        evidence: Option<&str>,
        functional: &str,
        regime: &str,
        graph: &AcceptedGraph,
    ) -> Result<crate::analysis::events::AcceptedExcitation, crate::analysis::events::EventError>
    {
        let ctx = ExecutionContext::for_tests(0);
        Study::prepare_excitation_question(
            reading,
            subject(alpha),
            bridge(functional, evidence, regime),
            graph,
            &ctx,
        )
        .unwrap()
        .execute(&ctx)
    }

    #[test]
    fn agreement_carries_the_direct_mass_when_discovery_orients_the_edge() {
        let ctx = ExecutionContext::for_tests(0);
        let observed = history();
        let reading = agreed(0.4);
        let graph = discovered("release", "activity");
        let accepted =
            ask(reading.clone(), 0.4, None, "direct_offspring", "do_one_release", &graph).unwrap();
        assert!((accepted.bridged_direct - 2.0 * reading.conditional_direct).abs() < 1e-12);
        assert!((accepted.lifetime_descendants - accepted.bridged_direct).abs() > 1e-3);
        let law = accepted
            .release_law(
                Continuation::Catalog(&observed),
                ReleaseAction::Now,
                ReleaseDraw { release: 0.0, window_end: 2.0, paths: 1, max_events: 4, seed: 7 },
                &ctx,
            )
            .unwrap();
        assert_eq!(law.meaning(), EventLawMeaning::Intervened);
        assert_eq!(law.paths()[0][0].time, 1.0);
        assert_eq!(law.paths()[0][0].kind, 0);
        let delayed = accepted
            .release_law(
                Continuation::Catalog(&observed),
                ReleaseAction::Delay { release: 1.5 },
                ReleaseDraw { release: 0.0, window_end: 2.0, paths: 1, max_events: 4, seed: 7 },
                &ctx,
            )
            .unwrap();
        assert_eq!(delayed.paths()[0][0].time, 1.5);
        let changed = accepted
            .release_law(
                Continuation::Catalog(&observed),
                ReleaseAction::ChangeMark { kind: 1, release: 1.2 },
                ReleaseDraw { release: 0.0, window_end: 2.0, paths: 1, max_events: 4, seed: 7 },
                &ctx,
            )
            .unwrap();
        assert_eq!(changed.paths()[0][0].kind, 1);
        let withheld = ExcitationReading { status: DiagnosticStatus::Withheld, ..reading };
        assert!(ask(withheld, 0.4, None, "direct_offspring", "do_one_release", &graph).is_err());
        let quiet = ask(agreed(0.0), 0.0, None, "direct_offspring", "do_one_release", &graph)
            .unwrap()
            .intervened_law(
                Continuation::Catalog(&observed),
                ReleaseDraw { release: 1.0, window_end: 2.0, paths: 1, max_events: 8, seed: 7 },
                &ctx,
            )
            .unwrap();
        assert_eq!(quiet.paths()[0][0], crate::analysis::events::Arrival { time: 1.0, kind: 0 });
    }

    #[test]
    fn a_label_a_reverse_edge_and_a_descendant_stay_out_of_the_bridge() {
        let reading = agreed(0.4);
        let graph = discovered("release", "activity");
        assert!(
            ask(reading.clone(), 0.4, None, "lifetime_descendants", "do_one_release", &graph)
                .is_err()
        );
        assert!(
            ask(
                reading.clone(),
                0.4,
                Some("kernel_audit"),
                "direct_offspring",
                "do_one_release",
                &graph
            )
            .is_err()
        );
        assert!(
            ask(reading.clone(), 0.4, None, "direct_offspring", "observational", &graph).is_err()
        );
        let reverse = discovered("activity", "release");
        assert!(
            ask(reading.clone(), 0.4, None, "direct_offspring", "do_one_release", &reverse)
                .is_err()
        );
        let mut predictive = bridge("direct_offspring", None, "do_one_release");
        predictive.kind = BridgeKind::PredictiveCausalQuantity;
        let ctx = ExecutionContext::for_tests(0);
        assert!(
            Study::prepare_excitation_question(reading, subject(0.4), predictive, &graph, &ctx)
                .unwrap()
                .execute(&ctx)
                .is_err()
        );
        let shock = Study::prepare_shock_query(
            super::ExcludedShock {
                history: history(),
                feature: BaselineFeature { knots: vec![0.0, 0.4], values: vec![0.0, 1.0] },
                source: 0,
                target: 1,
                horizon: 1.0,
                instrument: "launch_window".into(),
            },
            &ctx,
        )
        .unwrap()
        .execute(&ctx)
        .unwrap();
        assert_eq!(shock.exposure, "release");
        assert_eq!(shock.outcome, "activity");
        assert_eq!(shock.instrument, "launch_window");
    }

    #[test]
    fn a_marked_kernel_uses_the_same_question_and_one_added_prototype() {
        let ctx = ExecutionContext::for_tests(0);
        let prototype = MarkPrototype { source: vec![1.0], target: vec![1.0], mass: 1.0 };
        let other = prototype.clone();
        let history = MarkedHistory {
            time_unit: "hour".into(),
            origin: "origin".into(),
            source: "log".into(),
            empty_origin: 0.0,
            window: EventWindow { start: 0.0, end: 1.0 },
            left: vec![],
            events: vec![
                MarkedArrival { time: 0.2, prototype: 0 },
                MarkedArrival { time: 0.6, prototype: 1 },
            ],
            prototypes: vec![prototype.clone(), other.clone()],
        };
        let parameters =
            MarkedParameters { baseline: vec![0.0], excitation: vec![0.0], decay: vec![1.0] };
        let graph = discovered("prototype-0", "prototype-1");
        let accepted = Study::prepare_excitation_question(
            agreed(0.0),
            ExcitationSubject::Marked {
                parameters,
                prototypes: vec![prototype, other],
                history: history.clone(),
                reproduction_cap: 0.9,
            },
            bridge("direct_offspring", None, "do_one_release"),
            &graph,
            &ctx,
        )
        .unwrap()
        .execute(&ctx)
        .unwrap();
        let law = accepted
            .intervened_law(
                Continuation::Marked(&history),
                ReleaseDraw { release: 1.0, window_end: 2.0, paths: 1, max_events: 4, seed: 9 },
                &ctx,
            )
            .unwrap();
        assert_eq!(law.meaning(), EventLawMeaning::Intervened);
        assert_eq!(law.paths()[0][0], crate::analysis::events::Arrival { time: 1.0, kind: 0 });
    }
}

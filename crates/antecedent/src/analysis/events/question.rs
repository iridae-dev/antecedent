//! Submit a catalog or marked excitation mass to the parameter-intervention bridge.
//!
//! The immigrant diagnostic is a robustness check. This module does not treat
//! agreement as identification. An agreed direct mass can cross
//! [`BridgeKind::ParameterInterventionResponse`] read against a declared structure: an accepted
//! graph or equivalence class, a candidate set, a graph posterior or a cyclic system. Every
//! member of the structure is read: the kernel's mass where the member orients the exposure
//! into the outcome, zero where the edge is absent or reversed, nothing where the member does
//! not identify the effect (see [`super::discovery`]). The structure's provenance, or its
//! canonical identity, is the evidence. Descendants stay a conditional intensity under an
//! affine map: the resolvent is nonlinear, so the affine bridge carries the direct mass and the
//! resolvent descendants are reported beside it. Under an identity map the lifetime total
//! offspring (`total_offspring`: direct mass and resolvent feedback descendants) may cross as
//! well, so mutual excitation is read, not refused. A withheld diagnostic, a missing reading,
//! or a bridge that asks for anything else refuses, and no intervened law is produced for the
//! decision.

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
use antecedent_core::execution::AdmissionLease;
use antecedent_core::identity::{ContractIdentities, SemanticDigest};
use serde::Serialize;

use super::challenge::{DiagnosticStatus, ExcitationReading};
use super::discovery::{
    Mixture, ReadingMode, Structural, StructuralReading, has_feedback_member, has_null_member,
    provenance_hex, single_dag, structural_members,
};
use super::hawkes::{self, HawkesParameters};
use super::marked::{self, MarkPrototype, MarkedHistory, MarkedParameters};
use super::{
    Arrival, EventError, EventHistory, EventLaw, EventWindow, PreparedKernel, admit, digest,
};
use crate::planner::lowered::EventProcessRoute::ParameterBridge;
use antecedent_core::reason_code;
use antecedent_core::reason_details::events;

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

pub(super) enum Kernel {
    Catalog(HawkesParameters),
    Marked(MarkedParameters, Vec<MarkPrototype>),
}

/// A question whose direct mass the parameter bridge accepted.
///
/// `bridged_direct` is the bridge's mean of the kernel's reading for a member that orients the
/// exposure into the outcome; [`Self::structure`] reads every member of the declared structure.
/// The descendant fields are functionals of the conditional intensity and are not that mean.
pub struct AcceptedExcitation {
    /// Mapped kernel reading: the horizon direct mass, or under an identity map of
    /// `total_offspring` the lifetime total offspring.
    pub bridged_direct: f64,
    /// Constant-baseline horizon direct mass that the bridge carried.
    pub conditional_direct: f64,
    /// Filter-ODE descendants on the challenged horizon. Not a bridged mean.
    pub horizon_descendants: f64,
    /// Lifetime resolvent descendants, excluding the initiator. Not a bridged mean.
    pub lifetime_descendants: f64,
    /// The bridge carried evidence established in another scope, under a premise.
    pub transported: bool,
    /// The reading over every member of the declared structure: per-member bridged readings,
    /// their envelope, the mixture conditional on identified members where weights exist, the
    /// identified weight, and the resolvent descendants beside a direct mass when excitation
    /// runs both ways.
    pub structure: StructuralReading,
    pub(super) mixture: Mixture,
    pub(super) kernel: Kernel,
    pub(super) source: usize,
    pub(super) target: usize,
    pub(super) identity: SemanticDigest,
    _lease: AdmissionLease,
}

/// Compiled question, checked before the bridge derives a claim.
pub struct PreparedExcitationQuestion {
    reading: ExcitationReading,
    subject: ExcitationSubject,
    bridge: BridgeContract,
    structural: Structural,
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
    /// Unit of the offspring count.
    pub units: String,
    /// Population the quantity was read in.
    pub population_id: String,
    /// Regime of this endpoint.
    pub regime_id: String,
    /// Horizon the functional is taken over.
    pub horizon: u32,
    /// `direct_offspring`, or `total_offspring` under an identity map.
    pub functional_id: String,
}

/// Build the parameter-intervention contract the excitation question can offer.
///
/// Evidence is left empty. The declared structure writes its provenance digest, or its
/// canonical identity, in before the bridge is asked to carry the mean.
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
            return Err(EventError::bridge(
                events::INCOMPLETE_MAPPING,
                reason_code!("scientific_bridge_invalid"),
                "an affine map needs both an intercept and a slope",
            ));
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
    /// `structure` is any declared structure the shared member carrier takes: an accepted
    /// graph or equivalence class (a `&AcceptedGraph` converts), a candidate set, a graph
    /// posterior or a cyclic system. Its members are enumerated and identified here.
    ///
    /// # Errors
    /// The structure is not bound to the catalog names or no member identifies the reading;
    /// at execution, the diagnostic was withheld, the kernel does not reproduce it, the
    /// contract asks the bridge to carry a nonlinear descendant, or the bridge refuses the
    /// declaration.
    pub fn prepare_excitation_question(
        reading: ExcitationReading,
        subject: ExcitationSubject,
        bridge: BridgeContract,
        structure: impl Into<crate::analysis::structure_members::DeclaredStructure>,
        ctx: &ExecutionContext,
    ) -> Result<PreparedExcitationQuestion, EventError> {
        super::super::lowering::refuse_reentry(ctx)?;
        let lease = admit(ctx, 0, 8_192)?;
        let structure = structure.into();
        let (exposure, outcome) = pair_names(&subject, &reading)?;
        let structural = structural_members(&structure, &exposure, &outcome, ctx)?;
        let subject_hex = subject_identity(&subject)?.to_hex();
        let source = if single_dag(&structure) {
            digest(
                "antecedent.events.excitation_question.v2",
                &(&reading, subject_hex, bridge.identity().to_hex(), provenance_hex(&structure)),
            )?
        } else {
            digest(
                "antecedent.events.excitation_question.v3",
                &(
                    &reading,
                    subject_hex,
                    bridge.identity().to_hex(),
                    structural.members.identity().to_hex(),
                    provenance_hex(&structure),
                    ReadingMode::declared(&bridge).functional(),
                    super::discovery::CYCLIC.as_str(),
                ),
            )?
        };
        let plan = PreparedKernel::on(ParameterBridge, source, "excitation_question")?;
        Ok(PreparedExcitationQuestion { reading, subject, bridge, structural, plan, _lease: lease })
    }
}

impl PreparedExcitationQuestion {
    /// Inspect the compiled question.
    pub fn plan_view(&self) -> crate::planner::PlanView {
        self.plan.plan_view()
    }

    /// The members of the declared structure and their identification of the reading.
    #[must_use]
    pub fn members(&self) -> &crate::analysis::structure_members::StructureMembers {
        &self.structural.members
    }

    /// The evidence the bridge cites: the structure's provenance digest, or its canonical
    /// identity, as lowercase hex.
    #[must_use]
    pub fn evidence(&self) -> &str {
        &self.structural.evidence
    }

    /// Ask the parameter bridge to carry the agreed direct mass.
    ///
    /// # Errors
    /// The diagnostic, the kernel, or the bridge contract does not support
    /// the question. A refusal produces no claim and no intervened law.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<AcceptedExcitation, EventError> {
        let scoped = super::scoped_execution(ctx, &self.plan)?;
        accept(
            &self.reading,
            &self.subject,
            &self.bridge,
            &self.structural,
            self.plan.identity(),
            &scoped,
        )
    }
}

/// The canonical identity of the kernel a question offers: its coefficients, the history its
/// Gram and cap are checked against, and that cap.
fn subject_identity(subject: &ExcitationSubject) -> Result<SemanticDigest, EventError> {
    match subject {
        ExcitationSubject::Catalog { parameters, history, column_cap } => digest(
            "antecedent.events.excitation_subject.catalog.v1",
            &(parameters, history, column_cap),
        ),
        ExcitationSubject::Marked { parameters, prototypes, history, reproduction_cap } => digest(
            "antecedent.events.excitation_subject.marked.v1",
            &(parameters, prototypes, history, reproduction_cap),
        ),
    }
}

/// Offer the agreed kernel reading of `subject` to the parameter bridge under the
/// dispatch-scoped `ctx`, citing the declared structure as its structural evidence, and read
/// every member of that structure.
pub(super) fn accept(
    reading: &ExcitationReading,
    subject: &ExcitationSubject,
    bridge: &BridgeContract,
    structural: &Structural,
    identity: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<AcceptedExcitation, EventError> {
    let mass = offered_mass(reading, subject)?;
    let two_way = kernel_gate(subject, reading, ctx)?;
    let bridge = cite_discovery(bridge, &structural.evidence)?;
    let mode = direct_bridge(&bridge)?;
    let kernel_reading = match mode {
        ReadingMode::Direct => mass.direct,
        ReadingMode::Total => mass.lifetime_descendants,
    };
    let identities =
        ContractIdentities::new(identity, identity, None, None, identity, identity, identity);
    let derived = bridged(&bridge, &identities, kernel_reading)?;
    // A member where the edge is absent or reversed reads zero through the same bridge.
    let bridged_zero = if has_null_member(&structural.members) {
        bridged(&bridge, &identities, 0.0)?.value
    } else {
        derived.value
    };
    let feedback = two_way || has_feedback_member(&structural.members);
    let cyclic_reading =
        (feedback && mode == ReadingMode::Direct).then_some(mass.lifetime_descendants);
    let structure =
        super::discovery::read(structural, mode, derived.value, bridged_zero, cyclic_reading);
    Ok(AcceptedExcitation {
        bridged_direct: derived.value,
        conditional_direct: mass.direct,
        horizon_descendants: mass.horizon_descendants,
        lifetime_descendants: mass.lifetime_descendants,
        transported: derived.transported,
        structure,
        mixture: Mixture::of(&structural.members),
        kernel: kernel_of(subject),
        source: reading.source,
        target: reading.target,
        identity,
        _lease: admit(ctx, 0, 8_192)?,
    })
}

/// The bridge's derived claim for one kernel reading.
fn bridged(
    bridge: &BridgeContract,
    identities: &ContractIdentities,
    value: f64,
) -> Result<antecedent_contract::bridge::BridgedClaim, EventError> {
    let claim = point_claim(identities, &bridge.source.quantity, value);
    derive_bridged_claim(bridge, &BridgeSource { claim: &claim, quantity: &bridge.source.quantity })
        .map_err(|refusal| {
            EventError::bridge(
                events::BRIDGE_CONTRACT_REFUSED,
                reason_code!("scientific_bridge_invalid"),
                format!(
                    "{} {}: {}",
                    refusal.code,
                    refusal.detail,
                    refusal
                        .remedy
                        .unwrap_or("the bridge names the declaration that would support it")
                ),
            )
        })
}

impl AcceptedExcitation {
    /// Identity of the compiled program whose bridge accepted this excitation. A continuation
    /// or a release study composed on it names it as its parent.
    #[must_use]
    pub const fn identity(&self) -> SemanticDigest {
        self.identity
    }

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
    /// only because this question was accepted. This compiles and runs
    /// [`crate::Study::prepare_intervened_law`].
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
        crate::Study::prepare_intervened_law(self, continuation, draw, ctx)?.execute(ctx)
    }

    /// Draw one release action: now, later, or under a different mark.
    ///
    /// Each action is one added event. The excitation coefficients stay the
    /// accepted kernel. This compiles and runs [`crate::Study::prepare_release_law`].
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
        crate::Study::prepare_release_law(self, continuation, action, draw, ctx)?.execute(ctx)
    }
}

fn offered_mass(
    reading: &ExcitationReading,
    subject: &ExcitationSubject,
) -> Result<Mass, EventError> {
    if reading.status != DiagnosticStatus::Agreed {
        return Err(EventError::bridge(
            events::DIAGNOSTIC_WITHHELD,
            reason_code!("bridge_substitution_refused"),
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
        return Err(EventError::bridge(
            events::KERNEL_MISMATCH,
            reason_code!("bridge_binding_mismatch"),
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

/// Check the bridge asks for a reading it can carry, and say which.
fn direct_bridge(bridge: &BridgeContract) -> Result<ReadingMode, EventError> {
    if bridge.kind != BridgeKind::ParameterInterventionResponse
        || bridge.source.class != QuantityClass::ModelParameter
        || bridge.target.class != QuantityClass::InterventionResponse
        || bridge.source_value != ValueKind::Mean
        || bridge.target_value != ValueKind::Mean
    {
        return Err(EventError::bridge(
            events::NOT_PARAMETER_MEAN,
            reason_code!("bridge_substitution_refused"),
            "an excitation mass enters the decision loop only as a parameter-intervention mean",
        ));
    }
    let mode = ReadingMode::declared(bridge);
    let total = mode == ReadingMode::Total;
    if bridge.source.quantity.functional_id != mode.functional()
        || bridge.target.quantity.functional_id != mode.functional()
        || (total && !matches!(bridge.mapping, BridgeMapping::Identity))
    {
        return Err(EventError::bridge(
            events::DESCENDANT_NOT_CARRIED,
            reason_code!("bridge_mapping_not_supported"),
            "only the horizon direct mass is affine; the total offspring crosses an identity map \
             only, and other descendants stay a conditional intensity",
        ));
    }
    if !matches!(bridge.mapping, BridgeMapping::Identity | BridgeMapping::Affine { .. }) {
        return Err(EventError::bridge(
            events::MAPPING_NOT_AFFINE,
            reason_code!("bridge_mapping_not_supported"),
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
            return Err(EventError::bridge(
                events::EVIDENCE_MISSING,
                reason_code!("bridge_assumption_missing"),
                "a structural reading of the excitation mass must name its evidence",
            ));
        }
    }
    if bridge.source.quantity.regime_id != "observational"
        || bridge.source.quantity.regime_id == bridge.target.quantity.regime_id
    {
        return Err(EventError::bridge(
            events::REGIME_NOT_INTERVENED,
            reason_code!("bridge_binding_mismatch"),
            "the fitted mass is observational and the response must name a different regime",
        ));
    }
    Ok(mode)
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
        return Err(EventError::bridge(
            events::EVIDENCE_MISSING,
            reason_code!("bridge_assumption_missing"),
            "a structural reading of the excitation mass must name its evidence",
        ));
    };
    if reading.evidence.as_ref().is_some_and(|text| text != evidence) {
        return Err(EventError::bridge(
            events::EVIDENCE_NOT_DISCOVERY,
            reason_code!("bridge_assumption_missing"),
            "the structural evidence is the declared structure, not a separate label",
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

/// Check the kernel's Gram and stability cap, and say whether excitation runs both ways
/// between the questioned types. A two-way kernel is mutual excitation: it is read, not refused.
fn kernel_gate(
    subject: &ExcitationSubject,
    reading: &ExcitationReading,
    ctx: &ExecutionContext,
) -> Result<bool, EventError> {
    let mut work = hawkes::Work::new(ctx);
    match subject {
        ExcitationSubject::Catalog { parameters, history, column_cap } => {
            if !(0.0..1.0).contains(column_cap) {
                return Err(EventError::invalid("the column cap stays inside (0, 1)"));
            }
            if hawkes::singular_gram(history, &parameters.decay, &mut work)? {
                return Err(singular_gram());
            }
            let n = parameters.baseline.len();
            let column: f64 =
                (0..n).map(|target| parameters.branching[target * n + reading.source]).sum();
            if column >= column_cap - 1e-7 {
                return Err(EventError::bridge(
                    events::ON_STABILITY_CAP,
                    reason_code!("dynamics_not_identified"),
                    "the source column is on its stability cap",
                ));
            }
            Ok(reading.source != reading.target
                && edge(parameters, reading.source, reading.target) > 0.0
                && edge(parameters, reading.target, reading.source) > 0.0)
        }
        ExcitationSubject::Marked { parameters, prototypes, history, reproduction_cap } => {
            if !(0.0..1.0).contains(reproduction_cap) {
                return Err(EventError::invalid("the reproduction cap stays inside (0, 1)"));
            }
            if marked::singular_gram(history, &parameters.decay, &mut work)? {
                return Err(singular_gram());
            }
            let forward = lifetime_direct(parameters, prototypes, reading.source, reading.target)?;
            let reverse = lifetime_direct(parameters, prototypes, reading.target, reading.source)?;
            let offspring = parameters.excitation_offspring(prototypes, reading.source, 1.0)?;
            let column: f64 = offspring.lifetime_direct.iter().sum();
            if column >= reproduction_cap - 1e-7 {
                return Err(EventError::bridge(
                    events::ON_STABILITY_CAP,
                    reason_code!("dynamics_not_identified"),
                    "the source prototype is on its reproduction cap",
                ));
            }
            Ok(reading.source != reading.target && forward > 0.0 && reverse > 0.0)
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

fn singular_gram() -> EventError {
    EventError::bridge(
        events::SINGULAR_GRAM,
        reason_code!("dynamics_not_identified"),
        "the excitation Gram is singular",
    )
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

pub(super) fn release_window(draw: ReleaseDraw) -> Result<EventWindow, EventError> {
    let window = EventWindow { start: draw.release, end: draw.window_end };
    window.validate()?;
    Ok(window)
}

pub(super) fn before_release(history: &EventHistory, release: f64) -> Vec<Arrival> {
    history
        .left
        .iter()
        .chain(&history.events)
        .copied()
        .filter(|event| event.time < release)
        .collect()
}

pub(super) fn marked_before(history: &MarkedHistory, release: f64) -> Vec<marked::MarkedArrival> {
    history
        .left
        .iter()
        .chain(&history.events)
        .copied()
        .filter(|event| event.time < release)
        .collect()
}

pub(super) fn catalog_of(history: &MarkedHistory) -> EventHistory {
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
#[non_exhaustive]
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
        let source = digest(
            "antecedent.events.shock_query.v2",
            &(
                &shock.history,
                &shock.feature,
                shock.source,
                shock.target,
                shock.horizon,
                &shock.instrument,
            ),
        )?;
        let plan = PreparedKernel::on(ParameterBridge, source, "shock_query")?;
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
    use crate::analysis::events::EventError;
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
    use antecedent_core::reason_code;
    use antecedent_core::reason_details::events;
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

    /// The question refused under the registered bridge detail `slot` and reason code `code`.
    fn refused(
        result: Result<crate::analysis::events::AcceptedExcitation, EventError>,
        slot: &str,
        code: &str,
    ) {
        let Err(error) = result else {
            panic!("the bridge accepted a reading it should refuse under {slot}");
        };
        assert_eq!(error.detail(), Some(slot), "{error}");
        assert_eq!(error.reason_code(), Some(code), "{error}");
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
        refused(
            ask(withheld, 0.4, None, "direct_offspring", "do_one_release", &graph),
            events::DIAGNOSTIC_WITHHELD,
            reason_code!("bridge_substitution_refused"),
        );
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
    fn a_label_and_a_descendant_stay_out_of_the_bridge_and_a_reverse_edge_reads_zero() {
        let reading = agreed(0.4);
        let graph = discovered("release", "activity");
        refused(
            ask(reading.clone(), 0.4, None, "lifetime_descendants", "do_one_release", &graph),
            events::DESCENDANT_NOT_CARRIED,
            reason_code!("bridge_mapping_not_supported"),
        );
        refused(
            ask(
                reading.clone(),
                0.4,
                Some("kernel_audit"),
                "direct_offspring",
                "do_one_release",
                &graph,
            ),
            events::EVIDENCE_NOT_DISCOVERY,
            reason_code!("bridge_assumption_missing"),
        );
        refused(
            ask(reading.clone(), 0.4, None, "direct_offspring", "observational", &graph),
            events::REGIME_NOT_INTERVENED,
            reason_code!("bridge_binding_mismatch"),
        );
        // A discovery that orients activity into release is one member whose edge from the
        // exposure is reversed: the complete ID algorithm identifies the exposure's effect as
        // zero, so the structural reading is the bridged zero (slope 2, intercept 0) while the
        // bridge's mean of the kernel reading stays 2 alpha (1 - e^{-1}).
        let reverse = discovered("activity", "release");
        let accepted =
            ask(reading.clone(), 0.4, None, "direct_offspring", "do_one_release", &reverse)
                .unwrap();
        assert!((accepted.bridged_direct - 2.0 * reading.conditional_direct).abs() < 1e-12);
        assert_eq!(accepted.structure.envelope, [0.0, 0.0]);
        assert_eq!(accepted.structure.members.len(), 1);
        assert!(accepted.structure.members[0].relation.reverse());
        assert_eq!(accepted.structure.members[0].bridged, Some(0.0));
        let ctx = ExecutionContext::for_tests(0);
        let quiet = accepted
            .release_law(
                Continuation::Catalog(&history()),
                ReleaseAction::Now,
                ReleaseDraw { release: 0.0, window_end: 2.0, paths: 2, max_events: 8, seed: 7 },
                &ctx,
            )
            .unwrap();
        // Every member reads zero: the continuation is drawn without the added event.
        assert!(quiet.weights().is_none());
        let added = crate::analysis::events::Arrival { time: 1.0, kind: 0 };
        assert!(quiet.paths().iter().all(|path| path.first() != Some(&added)));
        let mut predictive = bridge("direct_offspring", None, "do_one_release");
        predictive.kind = BridgeKind::PredictiveCausalQuantity;
        refused(
            Study::prepare_excitation_question(reading, subject(0.4), predictive, &graph, &ctx)
                .unwrap()
                .execute(&ctx),
            events::NOT_PARAMETER_MEAN,
            reason_code!("bridge_substitution_refused"),
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

    #[test]
    fn release_law_counts_match_the_linear_intensity_means() {
        // Only a release (type 0) excites activity (type 1): the 1 <- 0 mass is alpha with decay
        // beta = 1, nothing excites a release, and both baselines are 0.2. On [1, 2) the release
        // added at the end of the observation contributes alpha (1 - e^{-beta}) activity, each
        // observed release at s contributes alpha (e^{-beta (1 - s)} - e^{-beta (2 - s)}), and a
        // baseline release at u contributes alpha (1 - e^{-beta (2 - u)}), whose mean over the
        // baseline rate 0.2 is 0.2 alpha (1 - (1 - e^{-beta}) / beta).
        let ctx = ExecutionContext::for_tests(0);
        let observed = history();
        let accepted = ask(
            agreed(0.4),
            0.4,
            None,
            "direct_offspring",
            "do_one_release",
            &discovered("release", "activity"),
        )
        .unwrap();
        let paths = 4000;
        let prepared = Study::prepare_release_law(
            &accepted,
            Continuation::Catalog(&observed),
            ReleaseAction::Now,
            ReleaseDraw { release: 0.0, window_end: 2.0, paths, max_events: 64, seed: 11 },
            &ctx,
        )
        .unwrap();
        assert_eq!(prepared.plan_view().route().as_str(), "HawkesParameterBridge");
        let law = prepared.execute(&ctx).unwrap();
        assert_eq!(law.meaning(), EventLawMeaning::Intervened);
        let (alpha, beta, baseline) = (0.4_f64, 1.0_f64, 0.2_f64);
        let mut counts = [0_usize; 2];
        for path in law.paths() {
            assert_eq!(path[0], crate::analysis::events::Arrival { time: 1.0, kind: 0 });
            for event in &path[1..] {
                counts[event.kind] += 1;
            }
        }
        let releases = observed
            .events
            .iter()
            .filter(|event| event.kind == 0)
            .map(|event| event.time)
            .chain([1.0]);
        let excited: f64 =
            releases.map(|s| alpha * ((-beta * (1.0 - s)).exp() - (-beta * (2.0 - s)).exp())).sum();
        let immigrant = baseline * alpha * (1.0 - (1.0 - (-beta).exp()) / beta);
        let activity = baseline + excited + immigrant;
        let mean = |kind: usize| counts[kind] as f64 / paths as f64;
        // Standard errors over 4000 paths are below 0.008 for releases and 0.018 for activity.
        assert!((mean(0) - baseline).abs() < 0.04, "releases {} against {baseline}", mean(0));
        assert!((mean(1) - activity).abs() < 0.09, "activity {} against {activity}", mean(1));
    }

    #[test]
    fn a_release_question_withholds_what_the_python_fixture_withholds() {
        // The challenge of python/tests/test_events.py, compiled as one release question: the
        // reading is withheld, so no claim and no intervened law are produced.
        let ctx = ExecutionContext::for_tests(0);
        let arrivals = [
            (0.10, 0),
            (0.16, 1),
            (0.22, 1),
            (0.50, 0),
            (0.56, 1),
            (0.62, 1),
            (0.80, 0),
            (0.86, 1),
            (0.92, 1),
        ];
        let observed = EventHistory {
            catalog: vec!["release".into(), "activity".into()],
            time_unit: "hour".into(),
            origin: "launch".into(),
            source: "log".into(),
            empty_origin: 0.0,
            window: EventWindow { start: 0.0, end: 1.0 },
            left: vec![],
            events: arrivals
                .iter()
                .map(|&(time, kind)| crate::analysis::events::Arrival { time, kind })
                .collect(),
            gaps: vec![],
        };
        let challenge = crate::analysis::events::ExcitationChallenge {
            history: observed,
            fit: crate::analysis::events::HawkesFitSpec {
                decay: vec![0.4, 1.7, 1.0, 0.3],
                column_cap: 0.9,
                iterations: 800,
            },
            alternative: BaselineFeature { knots: vec![0.0, 0.4], values: vec![0.0, 1.0] },
            source: 0,
            target: 1,
            horizon: 1.0,
            tolerance: 0.05,
        };
        let question = crate::analysis::events::ReleaseQuestion {
            challenge: crate::analysis::events::QuestionChallenge::Catalog(challenge),
            bridge: bridge("direct_offspring", None, "do_one_release"),
            action: ReleaseAction::Now,
            draw: ReleaseDraw { release: 0.0, window_end: 2.0, paths: 1, max_events: 4, seed: 7 },
        };
        let prepared =
            Study::prepare_release_question(question, &discovered("release", "activity"), &ctx)
                .unwrap();
        assert_eq!(prepared.plan_view().route().as_str(), "HawkesParameterBridge");
        assert_eq!(prepared.evidence().map(|digest| digest.len()), Some(64));
        let outcome = prepared.execute(&ctx).unwrap();
        assert_eq!(outcome.reading.status, DiagnosticStatus::Withheld);
        assert_eq!(outcome.reading.reason, Some("immigrant-alternative fit is weakly identified"));
        assert!(outcome.accepted.is_none());
        assert!(outcome.law.is_none());
    }
}

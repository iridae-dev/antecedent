//! Submit a catalog or marked excitation mass to the parameter-intervention bridge.
//!
//! The immigrant diagnostic is a robustness check. This module does not treat
//! agreement as identification. An agreed direct mass can cross
//! [`BridgeKind::ParameterInterventionResponse`] when the caller declares a
//! structural reading and names the evidence for it. Descendants stay a
//! conditional intensity: the resolvent is nonlinear, so the affine bridge
//! does not carry them as a mean. A withheld diagnostic, a missing reading, or
//! a bridge that asks for anything else refuses, and no intervened law is
//! produced for the decision.

use std::sync::Arc;

use antecedent_contract::bridge::{
    AssumptionKind, BridgeContract, BridgeKind, BridgeMapping, BridgeSource, QuantityClass,
    ValueKind, derive_bridged_claim,
};
use antecedent_contract::{
    ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus, ReasoningView, ScientificQuantity,
    SlotAvailability,
};
use antecedent_core::ExecutionContext;
use antecedent_core::execution::{AdmissionLease, RngFactory};
use antecedent_core::identity::{ContractIdentities, SemanticDigest};

use super::challenge::{DiagnosticStatus, ExcitationReading};
use super::hawkes::{self, HawkesParameters, INTERVENE};
use super::marked::{self, MarkPrototype, MarkedHistory, MarkedParameters};
use super::{
    Arrival, EventError, EventHistory, EventLaw, EventLawMeaning, EventWindow, PreparedKernel,
    admit, digest, path_bytes, poll, validate_budget,
};

/// The kernel whose agreed direct mass is offered to the bridge.
#[derive(Clone, Debug)]
pub enum ExcitationSubject {
    /// Finite catalog.
    Catalog(HawkesParameters),
    /// Finite-rank prototypes.
    Marked {
        /// Fitted coefficients.
        parameters: MarkedParameters,
        /// Prototypes the coefficients were fitted against.
        prototypes: Vec<MarkPrototype>,
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
    plan: PreparedKernel,
    _lease: AdmissionLease,
}

impl crate::Study {
    /// Compile an excitation question for the parameter-intervention bridge.
    ///
    /// # Errors
    /// The diagnostic was withheld, the kernel does not reproduce it, the
    /// contract asks the bridge to carry a nonlinear descendant, or the
    /// bridge refuses the declaration.
    pub fn prepare_excitation_question(
        reading: ExcitationReading,
        subject: ExcitationSubject,
        bridge: BridgeContract,
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
            ),
        )?;
        let plan = PreparedKernel::new(source, "excitation_question", super::CAPABILITY)?;
        Ok(PreparedExcitationQuestion { reading, subject, bridge, plan, _lease: lease })
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
        let _scoped = super::scoped_execution(ctx, self.plan.identity(), super::CAPABILITY)?;
        let mass = offered_mass(&self.reading, &self.subject)?;
        direct_bridge(&self.bridge)?;
        let identities = ContractIdentities::new(
            self.plan.identity(),
            self.plan.identity(),
            None,
            None,
            self.plan.identity(),
            self.plan.identity(),
            self.plan.identity(),
        );
        let claim = point_claim(&identities, &self.bridge.source.quantity, mass.direct);
        let derived = derive_bridged_claim(
            &self.bridge,
            &BridgeSource { claim: &claim, quantity: &self.bridge.source.quantity },
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
        if draw.release <= catalog.empty_origin {
            return Err(EventError::invalid("the added event must follow the empty origin"));
        }
        let mut work = hawkes::Work::new(ctx);
        let rng = RngFactory::from_seed(draw.seed);
        let mut paths = Vec::with_capacity(draw.paths);
        for index in 0..draw.paths {
            let mut path = vec![Arrival { time: draw.release, kind: self.source }];
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
                    prefix
                        .push(marked::MarkedArrival { time: draw.release, prototype: self.source });
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
        ExcitationSubject::Catalog(parameters) => {
            let offspring = parameters.excitation_offspring(reading.source, reading.horizon)?;
            Mass {
                direct: component(&offspring.direct, reading.target)?,
                horizon_descendants: component(&offspring.horizon_descendants, reading.target)?,
                lifetime_descendants: component(&offspring.lifetime_descendants, reading.target)?,
            }
        }
        ExcitationSubject::Marked { parameters, prototypes } => {
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
        ExcitationSubject::Catalog(parameters) => Kernel::Catalog(parameters.clone()),
        ExcitationSubject::Marked { parameters, prototypes } => {
            Kernel::Marked(parameters.clone(), prototypes.clone())
        }
    }
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

#[cfg(test)]
mod question_tests {
    use super::{Continuation, ExcitationSubject, ReleaseDraw};
    use crate::Study;
    use crate::analysis::events::challenge::{DiagnosticStatus, ExcitationReading};
    use crate::analysis::events::hawkes::HawkesParameters;
    use crate::analysis::events::marked::{MarkPrototype, MarkedHistory, MarkedParameters};
    use crate::analysis::events::{EventHistory, EventLawMeaning, EventWindow};
    use antecedent_contract::bridge::{
        AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind,
        BridgeMapping, BridgeScope, QuantityClass, ValueKind,
    };
    use antecedent_contract::{QuantityRole, ScientificQuantity};
    use antecedent_core::ExecutionContext;

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
            target: 0,
            horizon: 1.0,
            conditional_direct: direct,
            alternative_direct: direct,
            lifetime_direct: alpha,
            lifetime_descendants: alpha / (1.0 - alpha),
            horizon_descendants: 0.0,
        }
    }

    fn catalog() -> (EventHistory, HawkesParameters) {
        let history = EventHistory {
            catalog: vec!["release".into()],
            time_unit: "hour".into(),
            origin: "origin".into(),
            source: "log".into(),
            empty_origin: 0.0,
            window: EventWindow { start: 0.0, end: 1.0 },
            left: vec![],
            events: vec![],
            gaps: vec![],
        };
        let parameters =
            HawkesParameters { baseline: vec![0.0], branching: vec![0.4], decay: vec![1.0] };
        (history, parameters)
    }

    #[test]
    fn agreement_carries_only_the_direct_mass_and_a_withheld_diagnostic_refuses() {
        let ctx = ExecutionContext::for_tests(0);
        let (history, parameters) = catalog();
        let reading = agreed(0.4);
        let accepted = Study::prepare_excitation_question(
            reading.clone(),
            ExcitationSubject::Catalog(parameters.clone()),
            bridge("direct_offspring", Some("kernel_audit"), "do_one_release"),
            &ctx,
        )
        .unwrap()
        .execute(&ctx)
        .unwrap();
        assert!((accepted.bridged_direct - 2.0 * reading.conditional_direct).abs() < 1e-12);
        assert!((accepted.lifetime_descendants - accepted.bridged_direct).abs() > 1e-3);
        let law = accepted
            .intervened_law(
                Continuation::Catalog(&history),
                ReleaseDraw { release: 1.0, window_end: 2.0, paths: 1, max_events: 4, seed: 7 },
                &ctx,
            )
            .unwrap();
        assert_eq!(law.meaning(), EventLawMeaning::Intervened);
        assert_eq!(law.paths()[0][0].time, 1.0);
        assert_eq!(law.paths()[0][0].kind, 0);
        let withheld = ExcitationReading { status: DiagnosticStatus::Withheld, ..reading };
        assert!(
            Study::prepare_excitation_question(
                withheld,
                ExcitationSubject::Catalog(parameters),
                bridge("direct_offspring", Some("kernel_audit"), "do_one_release"),
                &ctx,
            )
            .unwrap()
            .execute(&ctx)
            .is_err()
        );
        let quiet =
            HawkesParameters { baseline: vec![0.0], branching: vec![0.0], decay: vec![1.0] };
        let empty = Study::prepare_excitation_question(
            agreed(0.0),
            ExcitationSubject::Catalog(quiet),
            bridge("direct_offspring", Some("kernel_audit"), "do_one_release"),
            &ctx,
        )
        .unwrap()
        .execute(&ctx)
        .unwrap()
        .intervened_law(
            Continuation::Catalog(&history),
            ReleaseDraw { release: 1.0, window_end: 2.0, paths: 1, max_events: 4, seed: 7 },
            &ctx,
        )
        .unwrap();
        assert_eq!(empty.paths(), &[vec![crate::analysis::events::Arrival { time: 1.0, kind: 0 }]]);
    }

    #[test]
    fn descendants_and_a_reading_without_evidence_stay_out_of_the_bridge() {
        let ctx = ExecutionContext::for_tests(0);
        let (_, parameters) = catalog();
        let reading = agreed(0.4);
        assert!(
            Study::prepare_excitation_question(
                reading.clone(),
                ExcitationSubject::Catalog(parameters.clone()),
                bridge("lifetime_descendants", Some("kernel_audit"), "do_one_release"),
                &ctx,
            )
            .unwrap()
            .execute(&ctx)
            .is_err()
        );
        assert!(
            Study::prepare_excitation_question(
                reading.clone(),
                ExcitationSubject::Catalog(parameters.clone()),
                bridge("direct_offspring", None, "do_one_release"),
                &ctx,
            )
            .unwrap()
            .execute(&ctx)
            .is_err()
        );
        assert!(
            Study::prepare_excitation_question(
                reading.clone(),
                ExcitationSubject::Catalog(parameters.clone()),
                bridge("direct_offspring", Some("kernel_audit"), "observational"),
                &ctx,
            )
            .unwrap()
            .execute(&ctx)
            .is_err()
        );
        let mut predictive = bridge("direct_offspring", Some("kernel_audit"), "do_one_release");
        predictive.kind = BridgeKind::PredictiveCausalQuantity;
        assert!(
            Study::prepare_excitation_question(
                reading,
                ExcitationSubject::Catalog(parameters),
                predictive,
                &ctx,
            )
            .unwrap()
            .execute(&ctx)
            .is_err()
        );
    }

    #[test]
    fn a_marked_kernel_uses_the_same_question_and_one_added_prototype() {
        let ctx = ExecutionContext::for_tests(0);
        let prototype = MarkPrototype { source: vec![1.0], target: vec![1.0], mass: 1.0 };
        let history = MarkedHistory {
            time_unit: "hour".into(),
            origin: "origin".into(),
            source: "log".into(),
            empty_origin: 0.0,
            window: EventWindow { start: 0.0, end: 1.0 },
            left: vec![],
            events: vec![],
            prototypes: vec![prototype.clone()],
        };
        let parameters =
            MarkedParameters { baseline: vec![0.0], excitation: vec![0.0], decay: vec![1.0] };
        let accepted = Study::prepare_excitation_question(
            agreed(0.0),
            ExcitationSubject::Marked { parameters, prototypes: vec![prototype] },
            bridge("direct_offspring", Some("mark_audit"), "do_one_release"),
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
        assert_eq!(law.paths(), &[vec![crate::analysis::events::Arrival { time: 1.0, kind: 0 }]]);
    }
}

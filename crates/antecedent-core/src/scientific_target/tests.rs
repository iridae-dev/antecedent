use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::{
    Conditioning, ContrastForm, ContrastTerm, CrossWorldPremise, DoseTimeSurface, GraphFamily,
    HistoryWindow, InferenceSelection, InitialLaw, InterventionContrast, InterventionWorld,
    MediationArms, MediationComponent, MediationTarget, ModelAssumptions, ObservationUnit,
    OutcomeQuantity, QuantityUnits, ScientificTarget, ScientificTargetError, ScientificWorldId,
    TargetForm, TimeFrame, WorldAction, WorldBinding, WorldCollection,
};
use crate::VariableId;
use crate::assumption::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, PriorAssumption,
};
use crate::intervention::{Intervention, TemporalPolicy};
use crate::query::TargetPopulation;
use crate::value::Value;

fn vid(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn world_id(raw: u32) -> ScientificWorldId {
    ScientificWorldId::from_raw(raw)
}

fn model(graph: GraphFamily) -> ModelAssumptions {
    ModelAssumptions::new(graph, AssumptionSet::new())
}

fn binding(population: TargetPopulation, conditioning: Conditioning) -> WorldBinding {
    WorldBinding::new(
        population,
        ObservationUnit::new("subject").unwrap(),
        OutcomeQuantity::mean(vid(1), QuantityUnits::dimensionless()).unwrap(),
        conditioning,
    )
    .unwrap()
}

fn binding_with(
    population: TargetPopulation,
    unit: &str,
    outcome: VariableId,
    units: &str,
    conditioning: Conditioning,
) -> WorldBinding {
    WorldBinding::new(
        population,
        ObservationUnit::new(unit).unwrap(),
        OutcomeQuantity::mean(outcome, QuantityUnits::new(units).unwrap()).unwrap(),
        conditioning,
    )
    .unwrap()
}

fn target(form: TargetForm, assumptions: ModelAssumptions) -> ScientificTarget {
    ScientificTarget::new(form, assumptions, InferenceSelection::Unspecified)
}

fn temporal_contrast(
    policy: TemporalPolicy,
    horizon: u32,
    history: HistoryWindow,
    population: TargetPopulation,
    conditioning: Conditioning,
    initial_law: InitialLaw,
) -> ScientificTarget {
    let time = TimeFrame::temporal(horizon, history, initial_law).unwrap();
    let shared = binding(population, conditioning);
    let treatment = vid(0);
    let reference = InterventionWorld::new(
        world_id(0),
        WorldAction::schedule(set_level(treatment, 0.0), policy.clone()).unwrap(),
        true,
        shared.clone(),
        time.clone(),
    )
    .unwrap();
    let active = InterventionWorld::new(
        world_id(1),
        WorldAction::schedule(set_level(treatment, 1.0), policy).unwrap(),
        false,
        shared,
        time,
    )
    .unwrap();
    let worlds = WorldCollection::new([reference, active]).unwrap();
    let contrast =
        InterventionContrast::difference(worlds, world_id(1), world_id(0), world_id(0)).unwrap();
    target(TargetForm::Contrast(contrast), model(GraphFamily::Dag))
}

fn pulse_contrast() -> ScientificTarget {
    temporal_contrast(
        TemporalPolicy::pulse(0),
        1,
        HistoryWindow::planner_default(),
        TargetPopulation::AllObserved,
        Conditioning::unconditional(),
        InitialLaw::named("observed_origin").unwrap(),
    )
}

fn static_contrast(population: TargetPopulation, conditioning: Conditioning) -> ScientificTarget {
    let time = TimeFrame::static_decision();
    let shared = binding(population, conditioning);
    let treatment = vid(0);
    let reference = InterventionWorld::new(
        world_id(0),
        WorldAction::at_decision([set_level(treatment, 0.0)]).unwrap(),
        true,
        shared.clone(),
        time.clone(),
    )
    .unwrap();
    let active = InterventionWorld::new(
        world_id(1),
        WorldAction::at_decision([set_level(treatment, 1.0)]).unwrap(),
        false,
        shared,
        time,
    )
    .unwrap();
    let worlds = WorldCollection::new([reference, active]).unwrap();
    let contrast =
        InterventionContrast::difference(worlds, world_id(1), world_id(0), world_id(0)).unwrap();
    target(TargetForm::Contrast(contrast), model(GraphFamily::Dag))
}

fn set_level(variable: VariableId, level: f64) -> Intervention {
    Intervention::set(variable, Value::f64(level))
}

fn digest_ne(left: &ScientificTarget, right: &ScientificTarget) {
    assert_ne!(left, right);
    assert_ne!(left.identity(), right.identity());
}

/// `PulseEffect` and `SustainedEffect` both use `temporal.linear.adjustment`.
#[test]
fn pulse_effect_differs_from_sustained_effect() {
    let pulse = pulse_contrast();
    let sustained = temporal_contrast(
        TemporalPolicy::sustained(0, 1),
        1,
        HistoryWindow::planner_default(),
        TargetPopulation::AllObserved,
        Conditioning::unconditional(),
        InitialLaw::named("observed_origin").unwrap(),
    );
    digest_ne(&pulse, &sustained);
}

/// `TemporalPolicy::is_single_step` is true for both a pulse and `Sustained { from: 0, until: 0 }`.
#[test]
fn single_step_sustained_differs_from_pulse() {
    let pulse = pulse_contrast();
    let sustained = temporal_contrast(
        TemporalPolicy::sustained(0, 0),
        1,
        HistoryWindow::planner_default(),
        TargetPopulation::AllObserved,
        Conditioning::unconditional(),
        InitialLaw::named("observed_origin").unwrap(),
    );
    digest_ne(&pulse, &sustained);
}

/// `TemporalConditionalPulse` and `ConditionalEffect` (`conditional.linear.adjustment`).
#[test]
fn conditional_pulse_differs_from_static_conditional_effect() {
    let modifiers = Conditioning::on([vid(2)]);
    let pulse = temporal_contrast(
        TemporalPolicy::pulse(0),
        1,
        HistoryWindow::planner_default(),
        TargetPopulation::AllObserved,
        modifiers.clone(),
        InitialLaw::named("observed_origin").unwrap(),
    );
    let static_conditional = static_contrast(TargetPopulation::AllObserved, modifiers);
    digest_ne(&pulse, &static_conditional);
}

/// Horizon 1 does not make a pulse into `AverageEffect`.
#[test]
fn horizon_one_pulse_differs_from_static_average_effect() {
    digest_ne(
        &pulse_contrast(),
        &static_contrast(TargetPopulation::AllObserved, Conditioning::unconditional()),
    );
}

#[test]
fn inference_label_does_not_change_identity() {
    let frequentist = pulse_contrast().with_inference(InferenceSelection::Frequentist);
    let bayesian = pulse_contrast().with_inference(InferenceSelection::Bayesian);
    assert_eq!(frequentist, bayesian);
    assert_eq!(frequentist.identity(), bayesian.identity());
    assert_ne!(frequentist.inference(), bayesian.inference());
    let mut left = DefaultHasher::new();
    let mut right = DefaultHasher::new();
    frequentist.hash(&mut left);
    bayesian.hash(&mut right);
    assert_eq!(left.finish(), right.finish());
}

#[test]
fn population_changes_digest() {
    let all = pulse_contrast();
    let treated = temporal_contrast(
        TemporalPolicy::pulse(0),
        1,
        HistoryWindow::planner_default(),
        TargetPopulation::Treated,
        Conditioning::unconditional(),
        InitialLaw::named("observed_origin").unwrap(),
    );
    digest_ne(&all, &treated);
}

#[test]
fn horizon_changes_digest() {
    let one = pulse_contrast();
    let two = temporal_contrast(
        TemporalPolicy::pulse(0),
        2,
        HistoryWindow::planner_default(),
        TargetPopulation::AllObserved,
        Conditioning::unconditional(),
        InitialLaw::named("observed_origin").unwrap(),
    );
    digest_ne(&one, &two);
}

#[test]
fn history_changes_digest() {
    let default_history = pulse_contrast();
    let explicit_zero = temporal_contrast(
        TemporalPolicy::pulse(0),
        1,
        HistoryWindow::max_lag(0),
        TargetPopulation::AllObserved,
        Conditioning::unconditional(),
        InitialLaw::named("observed_origin").unwrap(),
    );
    let one_lag = temporal_contrast(
        TemporalPolicy::pulse(0),
        1,
        HistoryWindow::max_lag(1),
        TargetPopulation::AllObserved,
        Conditioning::unconditional(),
        InitialLaw::named("observed_origin").unwrap(),
    );
    digest_ne(&default_history, &explicit_zero);
    digest_ne(&default_history, &one_lag);
    digest_ne(&explicit_zero, &one_lag);
}

#[test]
fn initial_law_changes_digest() {
    let origin = pulse_contrast();
    let equilibrium = temporal_contrast(
        TemporalPolicy::pulse(0),
        1,
        HistoryWindow::planner_default(),
        TargetPopulation::AllObserved,
        Conditioning::unconditional(),
        InitialLaw::named("equilibrium").unwrap(),
    );
    let contemporaneous = temporal_contrast(
        TemporalPolicy::pulse(0),
        1,
        HistoryWindow::planner_default(),
        TargetPopulation::AllObserved,
        Conditioning::unconditional(),
        InitialLaw::contemporaneous_observed(),
    );
    digest_ne(&origin, &equilibrium);
    digest_ne(&origin, &contemporaneous);
}

#[test]
fn worlds_must_share_population_outcome_units_and_conditioning() {
    let time = TimeFrame::static_decision();
    let action = |level: f64| WorldAction::at_decision([set_level(vid(0), level)]).unwrap();
    let pair = |left: WorldBinding, right: WorldBinding| {
        WorldCollection::new([
            InterventionWorld::new(world_id(0), action(0.0), true, left, time.clone()).unwrap(),
            InterventionWorld::new(world_id(1), action(1.0), false, right, time.clone()).unwrap(),
        ])
    };
    let base = binding_with(
        TargetPopulation::AllObserved,
        "subject",
        vid(1),
        "1",
        Conditioning::unconditional(),
    );
    let cases = [
        (
            "population",
            binding_with(
                TargetPopulation::Treated,
                "subject",
                vid(1),
                "1",
                Conditioning::unconditional(),
            ),
        ),
        (
            "unit",
            binding_with(
                TargetPopulation::AllObserved,
                "cluster",
                vid(1),
                "1",
                Conditioning::unconditional(),
            ),
        ),
        (
            "outcome",
            binding_with(
                TargetPopulation::AllObserved,
                "subject",
                vid(4),
                "1",
                Conditioning::unconditional(),
            ),
        ),
        (
            "units",
            binding_with(
                TargetPopulation::AllObserved,
                "subject",
                vid(1),
                "USD",
                Conditioning::unconditional(),
            ),
        ),
        (
            "conditioning",
            binding_with(
                TargetPopulation::AllObserved,
                "subject",
                vid(1),
                "1",
                Conditioning::on([vid(3)]),
            ),
        ),
    ];
    for (field, other) in cases {
        let err = pair(base.clone(), other).unwrap_err();
        assert_eq!(err, ScientificTargetError::IncompatibleBinding { field });
    }
}

fn joint_action(a_level: f64, b_level: f64) -> WorldAction {
    WorldAction::at_decision([set_level(vid(0), a_level), set_level(vid(2), b_level)]).unwrap()
}

fn interaction_world(id: u32, reference: bool, a_level: f64, b_level: f64) -> InterventionWorld {
    InterventionWorld::new(
        world_id(id),
        joint_action(a_level, b_level),
        reference,
        binding(TargetPopulation::AllObserved, Conditioning::unconditional()),
        TimeFrame::static_decision(),
    )
    .unwrap()
}

#[test]
fn interaction_uses_explicit_reference_world() {
    let worlds = WorldCollection::new([
        interaction_world(0, true, 0.0, 0.0),
        interaction_world(1, false, 1.0, 0.0),
        interaction_world(2, false, 0.0, 1.0),
        interaction_world(3, false, 1.0, 1.0),
    ])
    .unwrap();
    let contrast = InterventionContrast::interaction(
        worlds,
        world_id(3),
        world_id(1),
        world_id(2),
        world_id(0),
    )
    .unwrap();
    assert_eq!(
        contrast.terms(),
        vec![
            ContrastTerm { world: world_id(3), coefficient: 1.0 },
            ContrastTerm { world: world_id(1), coefficient: -1.0 },
            ContrastTerm { world: world_id(2), coefficient: -1.0 },
            ContrastTerm { world: world_id(0), coefficient: 1.0 },
        ]
    );
    assert!(matches!(contrast.form(), ContrastForm::Interaction { .. }));
    let unmarked = WorldCollection::new([
        interaction_world(0, false, 0.0, 0.0),
        interaction_world(1, false, 1.0, 0.0),
        interaction_world(2, false, 0.0, 1.0),
        interaction_world(3, false, 1.0, 1.0),
    ])
    .unwrap();
    let err = InterventionContrast::interaction(
        unmarked,
        world_id(3),
        world_id(1),
        world_id(2),
        world_id(0),
    )
    .unwrap_err();
    assert_eq!(err, ScientificTargetError::ReferenceNotMarked { id: world_id(0) });

    let difference_worlds = WorldCollection::new([
        interaction_world(0, true, 0.0, 0.0),
        interaction_world(1, false, 1.0, 0.0),
    ])
    .unwrap();
    let err =
        InterventionContrast::difference(difference_worlds, world_id(1), world_id(0), world_id(1))
            .unwrap_err();
    assert_eq!(err, ScientificTargetError::ReferenceIsNotBaseline { id: world_id(1) });
}

#[test]
fn dose_time_surface_is_a_grid_not_a_contrast() {
    let shared = binding(TargetPopulation::AllObserved, Conditioning::unconditional());
    let law = InitialLaw::named("observed_origin").unwrap();
    let mut built = Vec::new();
    let mut cells = Vec::new();
    for (dose_index, level) in [0.0_f64, 1.0].into_iter().enumerate() {
        for (horizon_index, horizon) in [1_u32, 2].into_iter().enumerate() {
            let id = world_id(u32::try_from(dose_index * 2 + horizon_index).unwrap());
            let time =
                TimeFrame::temporal(horizon, HistoryWindow::max_lag(1), law.clone()).unwrap();
            let action =
                WorldAction::schedule(set_level(vid(0), level), TemporalPolicy::pulse(0)).unwrap();
            built.push(
                InterventionWorld::new(id, action, id.raw() == 0, shared.clone(), time).unwrap(),
            );
            cells.push(id);
        }
    }
    let worlds = WorldCollection::new(built.clone()).unwrap();
    let surface = DoseTimeSurface::new(worlds.clone(), vid(0), [0.0, 1.0], [1, 2], cells).unwrap();
    assert_eq!(surface.horizons(), &[1, 2]);
    assert_eq!(surface.cells().len(), 4);
    let surface_target = target(TargetForm::DoseTime(surface), model(GraphFamily::Dag));
    assert!(matches!(surface_target.form(), TargetForm::DoseTime(_)));
    let bundle = target(TargetForm::Worlds(worlds), model(GraphFamily::Dag));
    digest_ne(&surface_target, &bundle);

    let reference = InterventionWorld::new(
        built[0].id(),
        built[0].action().clone(),
        true,
        shared.clone(),
        built[0].time().clone(),
    )
    .unwrap();
    let corners = WorldCollection::new([reference, built[3].clone()]).unwrap();
    let err = InterventionContrast::difference(corners, world_id(3), world_id(0), world_id(0))
        .unwrap_err();
    assert_eq!(err, ScientificTargetError::IncompatibleTime);

    let short = WorldCollection::new(built[..3].to_vec()).unwrap();
    let err = DoseTimeSurface::new(
        short,
        vid(0),
        [0.0, 1.0],
        [1, 2],
        [world_id(0), world_id(1), world_id(2)],
    )
    .unwrap_err();
    assert!(matches!(err, ScientificTargetError::NotADoseTimeGrid { .. }));
}

fn mediation(
    time: TimeFrame,
    schedule: Option<TemporalPolicy>,
    component: MediationComponent,
    premises: &[CrossWorldPremise],
) -> Result<MediationTarget, ScientificTargetError> {
    let arms = MediationArms::new(
        vid(0),
        [vid(2)],
        set_level(vid(0), 0.0),
        set_level(vid(0), 1.0),
        component,
    )
    .unwrap();
    MediationTarget::new(
        binding(TargetPopulation::AllObserved, Conditioning::unconditional()),
        time,
        schedule,
        arms,
        premises.iter().copied(),
    )
}

#[test]
fn mediation_requires_cross_world_premises_and_is_not_a_contrast() {
    let err = mediation(TimeFrame::static_decision(), None, MediationComponent::NaturalDirect, &[])
        .unwrap_err();
    assert_eq!(err, ScientificTargetError::MediationNeedsCrossWorldPremises);
    let mediated = mediation(
        TimeFrame::static_decision(),
        None,
        MediationComponent::NaturalDirect,
        &[
            CrossWorldPremise::Composition,
            CrossWorldPremise::CrossWorldIndependence,
            CrossWorldPremise::PathSpecificNested,
        ],
    )
    .unwrap();
    let contrast = static_contrast(TargetPopulation::AllObserved, Conditioning::unconditional());
    digest_ne(&target(TargetForm::Mediation(mediated), model(GraphFamily::Dag)), &contrast);
    assert!(matches!(contrast.form(), TargetForm::Contrast(_)));
}

/// `MediationEffect` and `TemporalMediationEffect`.
#[test]
fn temporal_mediation_differs_from_static_mediation() {
    let premises = [
        CrossWorldPremise::Composition,
        CrossWorldPremise::CrossWorldIndependence,
        CrossWorldPremise::PathSpecificNested,
    ];
    let static_mediation =
        mediation(TimeFrame::static_decision(), None, MediationComponent::NaturalDirect, &premises)
            .unwrap();
    let temporal_mediation = mediation(
        TimeFrame::temporal(
            1,
            HistoryWindow::planner_default(),
            InitialLaw::named("observed_origin").unwrap(),
        )
        .unwrap(),
        Some(TemporalPolicy::pulse(0)),
        MediationComponent::NaturalDirect,
        &premises,
    )
    .unwrap();
    digest_ne(
        &target(TargetForm::Mediation(static_mediation), model(GraphFamily::Dag)),
        &target(TargetForm::Mediation(temporal_mediation), model(GraphFamily::Dag)),
    );
}

#[test]
fn model_assumptions_change_digest() {
    let dag = pulse_contrast();
    let admg = ScientificTarget::new(
        dag.form().clone(),
        model(GraphFamily::Admg),
        InferenceSelection::Frequentist,
    );
    digest_ne(&dag, &admg);
    let mut records = AssumptionSet::new();
    records.push(AssumptionRecord {
        assumption: Assumption::PriorRestriction(PriorAssumption {
            id: Arc::from("half_normal"),
            description: Arc::from("shrinks an unidentified component"),
        }),
        source: AssumptionSource::UserDeclared,
        scope: AssumptionScope::Global,
        status: AssumptionStatus::Declared,
    });
    let with_prior = ScientificTarget::new(
        dag.form().clone(),
        ModelAssumptions::new(GraphFamily::Dag, records),
        InferenceSelection::Bayesian,
    );
    digest_ne(&dag, &with_prior);
}

fn interaction_of(
    reference: (f64, f64),
    a_at_reference: (f64, f64),
    b_at_reference: (f64, f64),
    joint: (f64, f64),
) -> Result<InterventionContrast, ScientificTargetError> {
    let worlds = WorldCollection::new([
        interaction_world(0, true, reference.0, reference.1),
        interaction_world(1, false, a_at_reference.0, a_at_reference.1),
        interaction_world(2, false, b_at_reference.0, b_at_reference.1),
        interaction_world(3, false, joint.0, joint.1),
    ])
    .unwrap();
    InterventionContrast::interaction(worlds, world_id(3), world_id(1), world_id(2), world_id(0))
}

#[test]
fn interaction_accepts_a_coherent_two_by_two_at_a_nonzero_reference() {
    // Y(a,b), Y(a,r_b), Y(r_a,b), Y(r_a,r_b) with a reference that is not zero.
    interaction_of((5.0, 5.0), (1.0, 5.0), (5.0, 2.0), (1.0, 2.0)).unwrap();
    interaction_of((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)).unwrap();
}

#[test]
fn interaction_refuses_worlds_that_do_not_realise_the_formula() {
    let cases = [
        // The old shifted reference: both factors differ from (5,5) at once.
        ((5.0, 5.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
        // The joint world takes the wrong level of A.
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 1.0)),
        // The joint world takes the wrong level of B.
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 3.0)),
        // a_at_reference moves B as well as A.
        ((0.0, 0.0), (1.0, 4.0), (0.0, 1.0), (1.0, 1.0)),
        // b_at_reference moves A as well as B.
        ((0.0, 0.0), (1.0, 0.0), (6.0, 1.0), (1.0, 1.0)),
        // Factor A never moves.
        ((0.0, 0.0), (0.0, 0.0), (0.0, 1.0), (0.0, 1.0)),
    ];
    for (reference, a, b, joint) in cases {
        let err = interaction_of(reference, a, b, joint).unwrap_err();
        assert!(
            matches!(err, ScientificTargetError::IncoherentContrast { .. }),
            "{reference:?} {a:?} {b:?} {joint:?}: {err:?}"
        );
    }
}

#[test]
fn interaction_projects_multi_channel_sequences_onto_their_own_coordinates() {
    use crate::intervention::{InterventionSequence, SequencedIntervention};
    let shared = binding(TargetPopulation::AllObserved, Conditioning::unconditional());
    let time =
        TimeFrame::temporal(2, HistoryWindow::max_lag(2), InitialLaw::contemporaneous_observed())
            .unwrap();
    let cell = |id: u32, a: [f64; 2], b: [f64; 2]| {
        let steps = (0..2)
            .flat_map(|index| {
                let at = i32::try_from(index).unwrap();
                [
                    SequencedIntervention::new(
                        set_level(vid(0), a[index]),
                        TemporalPolicy::pulse(at),
                    ),
                    SequencedIntervention::new(
                        set_level(vid(2), b[index]),
                        TemporalPolicy::pulse(at),
                    ),
                ]
            })
            .collect::<Vec<_>>();
        let action = WorldAction::schedule(
            Intervention::Sequence(InterventionSequence::new(steps)),
            TemporalPolicy::sustained(0, 1),
        )
        .unwrap();
        InterventionWorld::new(world_id(id), action, id == 0, shared.clone(), time.clone()).unwrap()
    };
    let build = |joint_b: [f64; 2], a_world_b: [f64; 2]| {
        let worlds = WorldCollection::new([
            cell(0, [0.0; 2], [0.0; 2]),
            cell(1, [1.0, 2.0], a_world_b),
            cell(2, [0.0; 2], [3.0, 4.0]),
            cell(3, [1.0, 2.0], joint_b),
        ])
        .unwrap();
        InterventionContrast::interaction(
            worlds,
            world_id(3),
            world_id(1),
            world_id(2),
            world_id(0),
        )
    };
    build([3.0, 4.0], [0.0; 2]).unwrap();
    assert!(matches!(
        build([3.0, 5.0], [0.0; 2]),
        Err(ScientificTargetError::IncoherentContrast { .. })
    ));
    assert!(matches!(
        build([3.0, 4.0], [0.0, 1.0]),
        Err(ScientificTargetError::IncoherentContrast { .. })
    ));
}

#[test]
fn interaction_accepts_a_dose_and_schedule_factorisation() {
    let shared = binding(TargetPopulation::AllObserved, Conditioning::unconditional());
    let time = TimeFrame::temporal(
        2,
        HistoryWindow::planner_default(),
        InitialLaw::contemporaneous_observed(),
    )
    .unwrap();
    let cell = |id: u32, reference: bool, level: f64, policy: TemporalPolicy| {
        InterventionWorld::new(
            world_id(id),
            WorldAction::schedule(set_level(vid(0), level), policy).unwrap(),
            reference,
            shared.clone(),
            time.clone(),
        )
        .unwrap()
    };
    let pulse = TemporalPolicy::pulse(0);
    let held = TemporalPolicy::sustained(0, 1);
    let build = |joint_policy: TemporalPolicy| {
        let worlds = WorldCollection::new([
            cell(0, true, 0.0, pulse.clone()),
            cell(1, false, 1.0, pulse.clone()),
            cell(2, false, 0.0, held.clone()),
            cell(3, false, 1.0, joint_policy),
        ])
        .unwrap();
        InterventionContrast::interaction(
            worlds,
            world_id(3),
            world_id(1),
            world_id(2),
            world_id(0),
        )
    };
    build(held.clone()).unwrap();
    let err = build(pulse.clone()).unwrap_err();
    assert!(matches!(err, ScientificTargetError::IncoherentContrast { .. }));
}

#[test]
fn difference_refuses_worlds_over_different_variables() {
    let one = WorldAction::at_decision([set_level(vid(0), 0.0)]).unwrap();
    let two = WorldAction::at_decision([set_level(vid(0), 1.0), set_level(vid(2), 1.0)]).unwrap();
    let shared = binding(TargetPopulation::AllObserved, Conditioning::unconditional());
    let time = TimeFrame::static_decision();
    let worlds = WorldCollection::new([
        InterventionWorld::new(world_id(0), one, true, shared.clone(), time.clone()).unwrap(),
        InterventionWorld::new(world_id(1), two, false, shared, time).unwrap(),
    ])
    .unwrap();
    let err = InterventionContrast::difference(worlds, world_id(1), world_id(0), world_id(0))
        .unwrap_err();
    assert!(matches!(err, ScientificTargetError::IncoherentContrast { .. }));
}

#[test]
fn symbolic_marker_is_not_a_world_level() {
    let marker = Intervention::set(vid(0), Value::symbolic_intervention());
    assert!(matches!(
        WorldAction::at_decision([marker.clone()]),
        Err(ScientificTargetError::InvalidAction { .. })
    ));
    assert!(matches!(
        WorldAction::schedule(marker.clone(), TemporalPolicy::pulse(0)),
        Err(ScientificTargetError::InvalidAction { .. })
    ));
    let shift = Intervention::shift(vid(0), Value::symbolic_intervention());
    assert!(WorldAction::at_decision([shift]).is_err());
    let err = MediationArms::new(
        vid(0),
        [vid(2)],
        set_level(vid(0), 0.0),
        marker,
        MediationComponent::Total,
    )
    .unwrap_err();
    assert!(matches!(err, ScientificTargetError::InvalidAction { .. }));
}

#[test]
fn mediation_arms_must_differ() {
    let err = MediationArms::new(
        vid(0),
        [vid(2)],
        set_level(vid(0), 1.0),
        set_level(vid(0), 1.0),
        MediationComponent::Total,
    )
    .unwrap_err();
    assert_eq!(err, ScientificTargetError::MediationArmsNotDistinct);
}

fn pulse_world(at: i32, horizon: u32) -> Result<InterventionWorld, ScientificTargetError> {
    InterventionWorld::new(
        world_id(0),
        WorldAction::schedule(set_level(vid(0), 1.0), TemporalPolicy::pulse(at)).unwrap(),
        false,
        binding(TargetPopulation::AllObserved, Conditioning::unconditional()),
        TimeFrame::temporal(
            horizon,
            HistoryWindow::planner_default(),
            InitialLaw::contemporaneous_observed(),
        )
        .unwrap(),
    )
}

#[test]
fn schedule_must_lie_within_the_horizon() {
    assert!(matches!(
        pulse_world(-3, 1),
        Err(ScientificTargetError::PolicyOutsideHorizon { horizon_steps: 1, .. })
    ));
    assert!(matches!(
        pulse_world(7, 1),
        Err(ScientificTargetError::PolicyOutsideHorizon { horizon_steps: 1, .. })
    ));
    pulse_world(0, 1).unwrap();
    pulse_world(-1, 1).unwrap();

    let window = |until: i32| {
        InterventionWorld::new(
            world_id(0),
            WorldAction::schedule(set_level(vid(0), 1.0), TemporalPolicy::sustained(0, until))
                .unwrap(),
            false,
            binding(TargetPopulation::AllObserved, Conditioning::unconditional()),
            TimeFrame::temporal(
                2,
                HistoryWindow::planner_default(),
                InitialLaw::contemporaneous_observed(),
            )
            .unwrap(),
        )
    };
    window(2).unwrap();
    assert!(matches!(window(3), Err(ScientificTargetError::PolicyOutsideHorizon { .. })));

    let premises = [
        CrossWorldPremise::Composition,
        CrossWorldPremise::CrossWorldIndependence,
        CrossWorldPremise::PathSpecificNested,
    ];
    let temporal = |at: i32| {
        mediation(
            TimeFrame::temporal(
                1,
                HistoryWindow::planner_default(),
                InitialLaw::contemporaneous_observed(),
            )
            .unwrap(),
            Some(TemporalPolicy::pulse(at)),
            MediationComponent::NaturalDirect,
            &premises,
        )
    };
    temporal(0).unwrap();
    assert!(matches!(temporal(7), Err(ScientificTargetError::PolicyOutsideHorizon { .. })));
    assert!(matches!(temporal(-3), Err(ScientificTargetError::PolicyOutsideHorizon { .. })));
}

#[test]
fn a_negative_pulse_inside_the_horizon_is_a_legitimate_target() {
    pulse_world(-1, 1).unwrap();
    assert!(matches!(
        pulse_world(-2, 1),
        Err(ScientificTargetError::PolicyOutsideHorizon { horizon_steps: 1, .. })
    ));
}

#[test]
fn natural_components_need_the_composition_premise() {
    for component in [MediationComponent::NaturalDirect, MediationComponent::NaturalIndirect] {
        let err = mediation(
            TimeFrame::static_decision(),
            None,
            component,
            &[CrossWorldPremise::CrossWorldIndependence, CrossWorldPremise::PathSpecificNested],
        )
        .unwrap_err();
        assert_eq!(err, ScientificTargetError::MediationNeedsCrossWorldPremises);
    }
    // Path-specific components are not defined through the nested counterfactual.
    mediation(
        TimeFrame::static_decision(),
        None,
        MediationComponent::Direct,
        &[CrossWorldPremise::CrossWorldIndependence, CrossWorldPremise::PathSpecificNested],
    )
    .unwrap();
}

fn record(
    assumption: Assumption,
    scope: AssumptionScope,
    status: AssumptionStatus,
) -> AssumptionRecord {
    AssumptionRecord { assumption, source: AssumptionSource::UserDeclared, scope, status }
}

fn with_records(records: Vec<AssumptionRecord>) -> ScientificTarget {
    let mut set = AssumptionSet::new();
    for entry in records {
        set.push(entry);
    }
    ScientificTarget::new(
        pulse_contrast().form().clone(),
        ModelAssumptions::new(GraphFamily::Dag, set),
        InferenceSelection::Unspecified,
    )
}

#[test]
fn assumption_identity_is_a_canonical_set() {
    let markov =
        || record(Assumption::CausalMarkov, AssumptionScope::Global, AssumptionStatus::Declared);
    let positivity =
        || record(Assumption::Positivity, AssumptionScope::Global, AssumptionStatus::Declared);
    let once = with_records(vec![markov()]).identity();
    assert_eq!(with_records(vec![markov(), markov()]).identity(), once);
    assert_eq!(
        with_records(vec![markov(), positivity()]).identity(),
        with_records(vec![positivity(), markov(), markov()]).identity()
    );
    // A diagnostic moving Declared to Supported, or a different source, is the same science.
    let supported =
        record(Assumption::CausalMarkov, AssumptionScope::Global, AssumptionStatus::Supported);
    assert_eq!(with_records(vec![supported]).identity(), once);
    let derived = AssumptionRecord {
        source: AssumptionSource::Derived { from: Arc::from("discovery") },
        ..markov()
    };
    assert_eq!(with_records(vec![derived]).identity(), once);
    // What is assumed, and where, is the science.
    assert_ne!(with_records(vec![positivity()]).identity(), once);
    let scoped =
        record(Assumption::CausalMarkov, AssumptionScope::Estimation, AssumptionStatus::Declared);
    assert_ne!(with_records(vec![scoped]).identity(), once);
    let custom = |description: &str| {
        record(
            Assumption::Custom { id: Arc::from("x"), description: Arc::from(description) },
            AssumptionScope::Global,
            AssumptionStatus::Declared,
        )
    };
    assert_ne!(
        with_records(vec![custom("a")]).identity(),
        with_records(vec![custom("b")]).identity()
    );
}

#[test]
fn contrast_identity_ignores_world_labels() {
    let relabelled = {
        let time = TimeFrame::static_decision();
        let shared = binding(TargetPopulation::AllObserved, Conditioning::unconditional());
        let reference = InterventionWorld::new(
            world_id(40),
            WorldAction::at_decision([set_level(vid(0), 0.0)]).unwrap(),
            true,
            shared.clone(),
            time.clone(),
        )
        .unwrap();
        let active = InterventionWorld::new(
            world_id(9),
            WorldAction::at_decision([set_level(vid(0), 1.0)]).unwrap(),
            false,
            shared,
            time,
        )
        .unwrap();
        let worlds = WorldCollection::new([reference, active]).unwrap();
        let contrast =
            InterventionContrast::difference(worlds, world_id(9), world_id(40), world_id(40))
                .unwrap();
        target(TargetForm::Contrast(contrast), model(GraphFamily::Dag))
    };
    let original = static_contrast(TargetPopulation::AllObserved, Conditioning::unconditional());
    assert_eq!(relabelled.identity(), original.identity());
}

fn static_world(id: u32, reference: bool, level: f64) -> InterventionWorld {
    InterventionWorld::new(
        world_id(id),
        WorldAction::at_decision([set_level(vid(0), level)]).unwrap(),
        reference,
        binding(TargetPopulation::AllObserved, Conditioning::unconditional()),
        TimeFrame::static_decision(),
    )
    .unwrap()
}

fn dose_surface(id_base: u32, high: f64, mark_first: bool) -> ScientificTarget {
    let shared = binding(TargetPopulation::AllObserved, Conditioning::unconditional());
    let law = InitialLaw::named("observed_origin").unwrap();
    let mut built = Vec::new();
    let mut cells = Vec::new();
    for (dose_index, level) in [0.0_f64, high].into_iter().enumerate() {
        for (horizon_index, horizon) in [1_u32, 2].into_iter().enumerate() {
            let slot = dose_index * 2 + horizon_index;
            let id = world_id(id_base + u32::try_from(slot).unwrap());
            let time =
                TimeFrame::temporal(horizon, HistoryWindow::max_lag(1), law.clone()).unwrap();
            let action =
                WorldAction::schedule(set_level(vid(0), level), TemporalPolicy::pulse(0)).unwrap();
            let reference = mark_first && slot == 0;
            built
                .push(InterventionWorld::new(id, action, reference, shared.clone(), time).unwrap());
            cells.push(id);
        }
    }
    let worlds = WorldCollection::new(built).unwrap();
    let surface = DoseTimeSurface::new(worlds, vid(0), [0.0, high], [1, 2], cells).unwrap();
    target(TargetForm::DoseTime(surface), model(GraphFamily::Dag))
}

#[test]
fn world_id_and_reference_flag_do_not_change_content_identity() {
    let assumptions = model(GraphFamily::Dag);
    let world = |id: u32, reference: bool, level: f64| {
        target(TargetForm::World(static_world(id, reference, level)), assumptions.clone())
    };
    let baseline = world(0, true, 0.0);
    let relabelled = world(7, false, 0.0);
    assert_ne!(baseline, relabelled);
    assert_eq!(baseline.identity(), relabelled.identity());
    assert_ne!(baseline.identity(), world(0, true, 1.0).identity());

    let bundle = |ids: [u32; 2], flags: [bool; 2], high: f64| {
        let worlds = WorldCollection::new([
            static_world(ids[0], flags[0], 0.0),
            static_world(ids[1], flags[1], high),
        ])
        .unwrap();
        target(TargetForm::Worlds(worlds), assumptions.clone())
    };
    let ordered = bundle([0, 1], [true, false], 1.0);
    let swapped = bundle([9, 4], [false, true], 1.0);
    assert_ne!(ordered, swapped);
    assert_eq!(ordered.identity(), swapped.identity());
    assert_ne!(ordered.identity(), bundle([0, 1], [true, false], 2.0).identity());

    let surface = dose_surface(0, 1.0, true);
    let relabelled_surface = dose_surface(10, 1.0, false);
    assert_ne!(surface, relabelled_surface);
    assert_eq!(surface.identity(), relabelled_surface.identity());
    assert_ne!(surface.identity(), dose_surface(0, 2.0, true).identity());
}

#[test]
fn negative_zero_level_is_the_same_world_as_zero() {
    let build = |level: f64| {
        let action = WorldAction::at_decision([set_level(vid(0), level)]).unwrap();
        InterventionWorld::new(
            world_id(0),
            action,
            true,
            binding(TargetPopulation::AllObserved, Conditioning::unconditional()),
            TimeFrame::static_decision(),
        )
        .unwrap()
    };
    let positive = target(TargetForm::World(build(0.0)), model(GraphFamily::Dag));
    let negative = target(TargetForm::World(build(-0.0)), model(GraphFamily::Dag));
    assert_eq!(positive, negative);
    assert_eq!(positive.identity(), negative.identity());
}

#[test]
fn contrast_form_is_not_mediation() {
    let contrast = static_contrast(TargetPopulation::AllObserved, Conditioning::unconditional());
    let TargetForm::Contrast(contrast) = contrast.form() else {
        panic!("static helper builds a contrast");
    };
    let label = match contrast.form() {
        ContrastForm::Linear => "linear",
        ContrastForm::Difference { .. } => "difference",
        ContrastForm::Interaction { .. } => "interaction",
    };
    assert_eq!(label, "difference");
}
fn three_factor_linear(id_offset: u32, reverse: bool) -> ScientificTarget {
    let binding = WorldBinding::new(
        TargetPopulation::AllObserved,
        ObservationUnit::new("group").unwrap(),
        OutcomeQuantity::mean(vid(4), QuantityUnits::new("USD").unwrap()).unwrap(),
        Conditioning::unconditional(),
    )
    .unwrap();
    let worlds = WorldCollection::new((0_u32..8).map(|cell| {
        InterventionWorld::new(
            world_id(id_offset + cell),
            WorldAction::at_decision(
                (0..3).map(|channel| set_level(vid(channel), f64::from((cell >> channel) & 1))),
            )
            .unwrap(),
            cell == 0,
            binding.clone(),
            TimeFrame::static_decision(),
        )
        .unwrap()
    }))
    .unwrap();
    let mut terms: Vec<_> = (0_u32..8)
        .map(|cell| {
            (world_id(id_offset + cell), if (3 - cell.count_ones()) % 2 == 0 { 1.0 } else { -1.0 })
        })
        .collect();
    if reverse {
        terms.reverse();
    }
    target(
        TargetForm::Contrast(
            InterventionContrast::linear(worlds, terms, world_id(id_offset)).unwrap(),
        ),
        model(GraphFamily::Dag),
    )
}
#[test]
fn arbitrary_linear_contrast_retains_higher_order_interaction_and_common_uncertainty_coefficients()
{
    let target = three_factor_linear(0, false);
    let TargetForm::Contrast(contrast) = target.form() else { panic!("linear target") };
    assert_eq!(contrast.form(), ContrastForm::Linear);
    assert_eq!(contrast.terms().len(), 8);
    let value = contrast
        .terms()
        .iter()
        .map(|term| {
            let cell = term.world.raw();
            let a = f64::from(cell & 1);
            let b = f64::from((cell >> 1) & 1);
            let c = f64::from((cell >> 2) & 1);
            term.coefficient * (a + b + c + 2.0 * a * b + 3.0 * a * b * c)
        })
        .sum::<f64>();
    assert!((value - 3.0).abs() < 1e-12);
    assert_eq!(target.identity(), three_factor_linear(0, true).identity());
    assert_eq!(target.identity(), three_factor_linear(100, true).identity());
}
#[test]
fn arbitrary_linear_contrast_refuses_missing_duplicated_or_nonfinite_terms() {
    let base = three_factor_linear(0, false);
    let TargetForm::Contrast(contrast) = base.form() else { panic!("linear target") };
    for invalid in [0.0, f64::NAN, f64::INFINITY] {
        let mut terms: Vec<_> =
            contrast.terms().iter().map(|term| (term.world, term.coefficient)).collect();
        terms[0].1 = invalid;
        assert!(
            InterventionContrast::linear(contrast.worlds().clone(), terms, world_id(0)).is_err()
        );
    }
    let mut terms: Vec<_> =
        contrast.terms().iter().map(|term| (term.world, term.coefficient)).collect();
    terms.pop();
    assert!(matches!(
        InterventionContrast::linear(contrast.worlds().clone(), terms, world_id(0)),
        Err(ScientificTargetError::ContrastWorldSet)
    ));
    let mut terms: Vec<_> =
        contrast.terms().iter().map(|term| (term.world, term.coefficient)).collect();
    terms[1].0 = terms[0].0;
    assert!(matches!(
        InterventionContrast::linear(contrast.worlds().clone(), terms, world_id(0)),
        Err(ScientificTargetError::WorldsNotDistinct)
    ));
}
#[test]
fn arbitrary_linear_contrast_supports_explicit_window_weights_and_identity_changes() {
    let binding = binding(TargetPopulation::AllObserved, Conditioning::unconditional());
    let worlds = WorldCollection::new((1..=3).map(|horizon| {
        InterventionWorld::new(
            world_id(horizon),
            WorldAction::schedule(set_level(vid(0), 1.0), TemporalPolicy::pulse(0)).unwrap(),
            horizon == 1,
            binding.clone(),
            TimeFrame::temporal(
                horizon,
                HistoryWindow::max_lag(2),
                InitialLaw::named("same history law").unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
    }))
    .unwrap();
    let mean = InterventionContrast::linear(
        worlds.clone(),
        [(world_id(1), 0.25), (world_id(2), 0.5), (world_id(3), 0.25)],
        world_id(1),
    )
    .unwrap();
    let changed = InterventionContrast::linear(
        worlds,
        [(world_id(1), 0.5), (world_id(2), 0.25), (world_id(3), 0.25)],
        world_id(1),
    )
    .unwrap();
    assert_ne!(
        target(TargetForm::Contrast(mean), model(GraphFamily::TemporalDag)).identity(),
        target(TargetForm::Contrast(changed), model(GraphFamily::TemporalDag)).identity()
    );
}

#[test]
fn mediation_refuses_conditioning_on_the_treatment_or_a_mediator() {
    for conditioned in [vid(0), vid(2)] {
        let arms = MediationArms::new(
            vid(0),
            [vid(2)],
            set_level(vid(0), 0.0),
            set_level(vid(0), 1.0),
            MediationComponent::Total,
        )
        .unwrap();
        let err = MediationTarget::new(
            binding(TargetPopulation::AllObserved, Conditioning::on([conditioned])),
            TimeFrame::static_decision(),
            None,
            arms,
            [],
        )
        .unwrap_err();
        let expected = if conditioned == vid(0) {
            ScientificTargetError::ConditioningIncludesTreatment { id: conditioned }
        } else {
            ScientificTargetError::ConditioningIncludesMediator { id: conditioned }
        };
        assert_eq!(err, expected);
    }
}

#[test]
fn a_resolved_planner_history_cap_is_pinned_by_identity() {
    let build = |history: HistoryWindow| {
        temporal_contrast(
            TemporalPolicy::pulse(0),
            2,
            history,
            TargetPopulation::AllObserved,
            Conditioning::unconditional(),
            InitialLaw::contemporaneous_observed(),
        )
    };
    let unresolved = build(HistoryWindow::planner_default());
    let two = build(HistoryWindow::planner_default().resolve(2));
    let three = build(HistoryWindow::planner_default().resolve(3));
    let declared = build(HistoryWindow::max_lag(2));
    digest_ne(&unresolved, &two);
    digest_ne(&two, &three);
    digest_ne(&two, &declared);
    assert_eq!(HistoryWindow::max_lag(4).resolve(2), HistoryWindow::max_lag(4));
    let resolved = HistoryWindow::planner_default().resolve(2);
    assert_eq!(resolved.resolved_cap(), Some(2));
    assert_eq!(resolved.max_lag_steps(), None);
    assert!(resolved.is_planner_default());
}

mod feedback_vocabulary {
    use super::super::StructuralAction;
    use super::*;

    fn cyclic_world(id: u32, action: WorldAction, time: TimeFrame) -> InterventionWorld {
        InterventionWorld::new(
            world_id(id),
            action,
            false,
            binding(TargetPopulation::AllObserved, Conditioning::unconditional()),
            time,
        )
        .unwrap()
    }

    fn cyclic_target(action: WorldAction, time: TimeFrame) -> ScientificTarget {
        target(TargetForm::World(cyclic_world(0, action, time)), model(GraphFamily::Cyclic))
    }

    fn one(action: StructuralAction) -> WorldAction {
        WorldAction::structural([action]).unwrap()
    }

    #[test]
    fn clamp_mechanism_shock_and_initial_state_are_distinct_worlds() {
        let continuous =
            || TimeFrame::continuous_time(2.0, InitialLaw::named("x0").unwrap()).unwrap();
        let clamp =
            cyclic_target(one(StructuralAction::hard_clamp(vid(0), 1.0).unwrap()), continuous());
        let mechanism = cyclic_target(
            one(StructuralAction::replace_mechanism(vid(0), "constant", &[1.0]).unwrap()),
            continuous(),
        );
        let shock = cyclic_target(
            one(StructuralAction::additive_shock(vid(0), 1.0).unwrap()),
            continuous(),
        );
        let initial = cyclic_target(
            one(StructuralAction::initial_state(vid(0), "x0", 1.0).unwrap()),
            continuous(),
        );
        let targets = [&clamp, &mechanism, &shock, &initial];
        for (index, left) in targets.iter().enumerate() {
            for right in &targets[index + 1..] {
                digest_ne(left, right);
            }
        }
    }

    #[test]
    fn equilibrium_stationary_continuous_and_static_frames_differ() {
        let clamp = || one(StructuralAction::hard_clamp(vid(0), 1.0).unwrap());
        let equilibrium = cyclic_target(clamp(), TimeFrame::equilibrium());
        let stationary = cyclic_target(clamp(), TimeFrame::stationary());
        let continuous = cyclic_target(
            clamp(),
            TimeFrame::continuous_time(1.0, InitialLaw::contemporaneous_observed()).unwrap(),
        );
        let longer = cyclic_target(
            clamp(),
            TimeFrame::continuous_time(2.0, InitialLaw::contemporaneous_observed()).unwrap(),
        );
        digest_ne(&equilibrium, &stationary);
        digest_ne(&equilibrium, &continuous);
        digest_ne(&continuous, &longer);
        let acyclic = target(
            TargetForm::World(cyclic_world(
                0,
                WorldAction::at_decision([set_level(vid(0), 1.0)]).unwrap(),
                TimeFrame::static_decision(),
            )),
            model(GraphFamily::Cyclic),
        );
        digest_ne(&equilibrium, &acyclic);
        assert!(TimeFrame::equilibrium().is_equilibrium());
        assert!(TimeFrame::stationary().is_stationary());
        assert!(!TimeFrame::equilibrium().is_static());
        assert_eq!(TimeFrame::equilibrium().horizon_steps(), None);
        assert_eq!(TimeFrame::equilibrium().initial_law(), None);
        let frame = TimeFrame::continuous_time(1.5, InitialLaw::named("x0").unwrap()).unwrap();
        assert_eq!(frame.continuous_horizon(), Some(1.5));
        assert_eq!(frame.continuous_initial_law().and_then(InitialLaw::name), Some("x0"));
        assert_eq!(frame.initial_law(), None, "a discrete-time accessor does not read it");
    }

    #[test]
    fn cyclic_graph_family_has_its_own_identity() {
        let action = || one(StructuralAction::hard_clamp(vid(0), 1.0).unwrap());
        let cyclic = cyclic_target(action(), TimeFrame::equilibrium());
        let unspecified = target(
            TargetForm::World(cyclic_world(0, action(), TimeFrame::equilibrium())),
            model(GraphFamily::Unspecified),
        );
        digest_ne(&cyclic, &unspecified);
    }

    #[test]
    fn continuous_horizon_must_be_finite_and_positive() {
        for horizon in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                TimeFrame::continuous_time(horizon, InitialLaw::contemporaneous_observed()),
                Err(ScientificTargetError::NonPositiveContinuousHorizon),
                "{horizon}"
            );
        }
    }

    #[test]
    fn structural_actions_pair_only_with_feedback_frames() {
        let clamp = || one(StructuralAction::hard_clamp(vid(0), 1.0).unwrap());
        let binding = binding(TargetPopulation::AllObserved, Conditioning::unconditional());
        for time in [
            TimeFrame::static_decision(),
            TimeFrame::temporal(
                1,
                HistoryWindow::planner_default(),
                InitialLaw::contemporaneous_observed(),
            )
            .unwrap(),
        ] {
            assert_eq!(
                InterventionWorld::new(world_id(0), clamp(), false, binding.clone(), time),
                Err(ScientificTargetError::MismatchedActionTime)
            );
        }
        let initial = one(StructuralAction::initial_state(vid(0), "x0", 1.0).unwrap());
        for time in [TimeFrame::equilibrium(), TimeFrame::stationary()] {
            assert_eq!(
                InterventionWorld::new(world_id(0), initial.clone(), false, binding.clone(), time),
                Err(ScientificTargetError::MismatchedActionTime),
                "an initial-state change needs a continuous-time frame"
            );
        }
        assert_eq!(
            InterventionWorld::new(
                world_id(0),
                WorldAction::at_decision([set_level(vid(0), 1.0)]).unwrap(),
                false,
                binding,
                TimeFrame::equilibrium(),
            ),
            Err(ScientificTargetError::MismatchedActionTime),
            "an acyclic decision-time intervention is not a structural action"
        );
    }

    #[test]
    fn structural_actions_are_canonical_and_refuse_conflicts() {
        let shock = StructuralAction::additive_shock(vid(1), 0.5).unwrap();
        let mechanism =
            StructuralAction::replace_mechanism(vid(1), "additive_shift", &[2.0]).unwrap();
        let clamp = StructuralAction::hard_clamp(vid(0), -0.0).unwrap();
        let forward =
            WorldAction::structural([shock.clone(), mechanism.clone(), clamp.clone()]).unwrap();
        let backward =
            WorldAction::structural([clamp.clone(), mechanism.clone(), shock.clone()]).unwrap();
        assert_eq!(forward, backward);
        assert_eq!(clamp.hard_clamp_level().map(f64::to_bits), Some(0.0_f64.to_bits()));
        assert_eq!(
            WorldAction::structural([shock.clone(), shock.clone()]),
            Err(ScientificTargetError::DuplicateInterventionVariable { id: vid(1) })
        );
        assert_eq!(
            WorldAction::structural([
                StructuralAction::hard_clamp(vid(1), 1.0).unwrap(),
                shock.clone()
            ]),
            Err(ScientificTargetError::DuplicateInterventionVariable { id: vid(1) })
        );
        assert!(StructuralAction::hard_clamp(vid(0), f64::NAN).is_err());
        assert!(StructuralAction::replace_mechanism(vid(0), " ", &[1.0]).is_err());
        assert!(StructuralAction::replace_mechanism(vid(0), "constant", &[f64::INFINITY]).is_err());
        assert!(StructuralAction::initial_state(vid(0), "", 1.0).is_err());
        let specified = WorldAction::structural([]).unwrap();
        assert_eq!(specified.structural_actions(), Some(&[][..]));
        assert!(specified.at_decision_interventions().is_none());
        assert!(specified.schedule_parts().is_none());
        let as_specified = cyclic_target(specified, TimeFrame::equilibrium());
        let clamped = cyclic_target(one(clamp), TimeFrame::equilibrium());
        digest_ne(&as_specified, &clamped);
    }

    #[test]
    fn structural_worlds_form_a_difference_on_one_coordinate() {
        let world = |id: u32, level: f64, reference: bool| {
            InterventionWorld::new(
                world_id(id),
                one(StructuralAction::hard_clamp(vid(0), level).unwrap()),
                reference,
                binding(TargetPopulation::AllObserved, Conditioning::unconditional()),
                TimeFrame::equilibrium(),
            )
            .unwrap()
        };
        let worlds = WorldCollection::new([world(0, 0.0, true), world(1, 1.0, false)]).unwrap();
        assert!(
            InterventionContrast::difference(worlds, world_id(1), world_id(0), world_id(0)).is_ok()
        );
        let shock = InterventionWorld::new(
            world_id(1),
            one(StructuralAction::additive_shock(vid(0), 1.0).unwrap()),
            false,
            binding(TargetPopulation::AllObserved, Conditioning::unconditional()),
            TimeFrame::equilibrium(),
        )
        .unwrap();
        let mixed = WorldCollection::new([world(0, 0.0, true), shock]).unwrap();
        assert!(
            InterventionContrast::difference(mixed, world_id(1), world_id(0), world_id(0)).is_err(),
            "a clamp and a shock are different coordinates, not two levels of one"
        );
    }
}

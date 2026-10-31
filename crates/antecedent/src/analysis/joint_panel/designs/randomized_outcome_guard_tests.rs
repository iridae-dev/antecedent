//! Regression for static hard outcome assignments in randomized world-mean compilation.
use super::*;
use antecedent_core::{
    Conditioning, GraphFamily, HistoryWindow, InferenceSelection, InitialLaw, Intervention,
    InterventionSequence, InterventionWorld, ModelAssumptions, ObservationUnit, OutcomeQuantity,
    QuantityUnits, ScientificWorldId, SequencedIntervention, TargetPopulation, TemporalPolicy,
    TimeFrame, Value, WorldAction, WorldBinding, WorldCollection, assumption::AssumptionSet,
};
use antecedent_estimate::joint_design::randomized::{
    JointRandomizedDesign, JointRandomizedUnit, randomized_joint_means,
};

fn worlds(action_variable: VariableId, conditioning: Conditioning) -> Vec<InterventionWorld> {
    let binding = WorldBinding::new(
        TargetPopulation::AllObserved,
        ObservationUnit::new("participant").unwrap(),
        OutcomeQuantity::mean(VariableId::from_raw(1), QuantityUnits::dimensionless()).unwrap(),
        conditioning,
    )
    .unwrap();
    (0..2)
        .map(|i| {
            InterventionWorld::new(
                ScientificWorldId::from_raw(i),
                WorldAction::at_decision([Intervention::set(
                    action_variable,
                    Value::f64(f64::from(i + 2)),
                )])
                .unwrap(),
                i == 0,
                binding.clone(),
                TimeFrame::static_decision(),
            )
            .unwrap()
        })
        .collect()
}

fn randomized(worlds: &[InterventionWorld]) -> (ScientificTarget, JointDesignProof) {
    let source_binding = JointDesignSourceBinding::from_world(&worlds[0], "target".into()).unwrap();
    let target = ScientificTarget::new(
        TargetForm::Worlds(WorldCollection::new(worlds.to_vec()).unwrap()),
        ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
        InferenceSelection::Frequentist,
    );
    let design = JointRandomizedDesign {
        worlds: vec!["arm0".into(), "arm1".into()],
        sampling_unit: "participant".into(),
        population: "target".into(),
        consistency: "assignment represents the declared world".into(),
        between_unit_isolation: "independent randomized units".into(),
    };
    let units = (0..4)
        .map(|i| JointRandomizedUnit {
            id: format!("unit{i}"),
            arm: i % 2,
            probabilities: vec![0.5, 0.5],
            outcome: 0.0,
            population_weight: 1.0,
        })
        .collect::<Vec<_>>();
    // Without the route guard, the source-data estimator returns zero for both
    // arm means even when the target worlds hard-set the outcome to 2 and 3.
    assert_eq!(randomized_joint_means(&design, &units).unwrap().means, vec![0.0, 0.0]);
    let proof = JointDesignProof {
        version: 1,
        source_binding,
        recipe: JointDesignRecipe::Randomized { design, units },
        worlds: worlds
            .iter()
            .enumerate()
            .map(|(i, world)| JointDesignWorldBinding {
                key: format!("arm{i}"),
                world: world.id().raw(),
                target_identity: joint_design_world_identity(world).to_hex(),
            })
            .collect(),
        longitudinal_arms: vec![],
    };
    (target, proof)
}

#[test]
fn randomized_world_mean_refuses_static_hard_outcome_assignments() {
    let ctx = ExecutionContext::for_tests(52);
    let ordinary = worlds(VariableId::from_raw(0), Conditioning::unconditional());
    let (target, proof) = randomized(&ordinary);
    let prepared = PreparedJointDesign::compile(target, proof, &ctx).unwrap();
    let fitted = prepared.execute(JointDesignInference::Frequentist, &ctx).unwrap();
    assert_eq!(fitted.law().estimate().means, vec![0.0, 0.0]);
    let hard_outcome = worlds(VariableId::from_raw(1), Conditioning::unconditional());
    let (target, proof) = randomized(&hard_outcome);
    let result = PreparedJointDesign::compile(target, proof, &ctx);
    assert!(
        matches!(result, Err(JointPanelError::Unsupported(message)) if message.contains("static hard assignment"))
    );
    let nested =
        Intervention::sequence(InterventionSequence::new(vec![SequencedIntervention::new(
            Intervention::set(VariableId::from_raw(1), Value::f64(2.0)),
            TemporalPolicy::pulse(0),
        )]));
    let nested_world = InterventionWorld::new(
        ScientificWorldId::from_raw(0),
        WorldAction::at_decision([nested]).unwrap(),
        false,
        hard_outcome[0].binding().clone(),
        TimeFrame::static_decision(),
    )
    .unwrap();
    assert!(matches!(
        reject_randomized_static_outcome_clamps(&[nested_world], &ctx),
        Err(JointPanelError::Unsupported(_))
    ));
}

#[test]
fn temporal_past_outcome_action_is_not_rejected_by_static_guard() {
    let binding =
        worlds(VariableId::from_raw(0), Conditioning::unconditional())[0].binding().clone();
    let world = InterventionWorld::new(
        ScientificWorldId::from_raw(0),
        WorldAction::schedule(
            Intervention::set(VariableId::from_raw(1), Value::f64(2.0)),
            TemporalPolicy::pulse(0),
        )
        .unwrap(),
        false,
        binding,
        TimeFrame::temporal(
            2,
            HistoryWindow::planner_default(),
            InitialLaw::contemporaneous_observed(),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(
        reject_randomized_static_outcome_clamps(&[world], &ExecutionContext::for_tests(52)).is_ok()
    );
}

#[test]
fn static_outcome_scan_bounds_nesting_and_honors_cancellation() {
    let binding =
        worlds(VariableId::from_raw(0), Conditioning::unconditional())[0].binding().clone();
    let mut intervention = Intervention::set(VariableId::from_raw(0), Value::f64(1.0));
    for _ in 0..66 {
        intervention =
            Intervention::sequence(InterventionSequence::new(vec![SequencedIntervention::new(
                intervention,
                TemporalPolicy::pulse(0),
            )]));
    }
    let world = InterventionWorld::new(
        ScientificWorldId::from_raw(0),
        WorldAction::at_decision([intervention]).unwrap(),
        false,
        binding,
        TimeFrame::static_decision(),
    )
    .unwrap();
    let ctx = ExecutionContext::for_tests(52);
    assert!(matches!(reject_randomized_static_outcome_clamps(std::slice::from_ref(&world), &ctx),
        Err(JointPanelError::Unsupported(message)) if message.contains("nesting depth")));
    ctx.cancellation.cancel();
    assert!(matches!(
        reject_randomized_static_outcome_clamps(&[world], &ctx),
        Err(JointPanelError::Engine(_))
    ));
}

#[test]
fn randomized_route_rejects_conditioned_targets_even_when_action_overlaps_conditioning() {
    let worlds = worlds(VariableId::from_raw(0), Conditioning::on([VariableId::from_raw(0)]));
    let (target, proof) = randomized(&worlds);
    let result = PreparedJointDesign::compile(target, proof, &ExecutionContext::for_tests(52));
    assert!(
        matches!(result, Err(JointPanelError::Compile(message)) if message.contains("conditioning"))
    );
}

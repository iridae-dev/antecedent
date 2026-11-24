//! The design-law reference decomposition is the same computation as the engine functions it
//! calls: `decompose_hybrid_law` on the realizations of a bootstrap design law, and
//! `propagate_hybrid_covariance` on the covariance of a posterior design law. Both are reached
//! through the compiled `JointDesign` route (`PreparedJointDesign::execute`, then
//! `JointDesignResult::decompose_reference_components`). Each test calls the engine function
//! on exactly the inputs the route hands it and compares every float through its round-trip
//! `Debug` form.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AssumptionSet, Conditioning, ExecutionContext, GraphFamily, InferenceSelection, Intervention,
    InterventionSequence, InterventionWorld, ModelAssumptions, ObservationUnit, OutcomeQuantity,
    QuantityUnits, ScientificTarget, ScientificWorldId, SequencedIntervention, TargetForm,
    TargetPopulation, TemporalPolicy, TimeFrame, Value, VariableId, WorldAction, WorldBinding,
    WorldCollection,
};
use antecedent_design::scientific_source::ScientificConsumptionScope;
use antecedent_estimate::joint_design::compiled::{JointDesignInference, JointDesignRecipe};

use super::{ChangeComponent, ReferenceComponentBindings};
use crate::analysis::joint_panel::designs::{
    JointDesignProof, JointDesignSourceBinding, JointDesignWorldBinding, PreparedJointDesign,
    gaussian_longitudinal_chain_fixture, joint_design_world_identity,
};
use antecedent_estimate::joint_panel as engine;

fn same<T: std::fmt::Debug + ?Sized>(label: &str, direct: &T, routed: &T) {
    assert_eq!(format!("{direct:?}"), format!("{routed:?}"), "{label}");
}

/// The continuous-history design compiled over its four worlds, the exposure bindings of
/// world 0 against world 3, and the bootstrap and posterior inferences.
fn design() -> (PreparedJointDesign, ReferenceComponentBindings, [JointDesignInference; 2]) {
    let ctx = ExecutionContext::for_tests(559);
    let (recipe, bayesian, _) = gaussian_longitudinal_chain_fixture();
    let JointDesignRecipe::GaussianLongitudinal { design, .. } = &recipe else {
        panic!("continuous history owner")
    };
    let worlds = design
        .worlds
        .iter()
        .enumerate()
        .map(|(index, program)| {
            let steps: Vec<_> = program
                .treatments
                .iter()
                .enumerate()
                .map(|(time, values)| {
                    SequencedIntervention::new(
                        Intervention::set(VariableId::from_raw(1), Value::f64(values[0])),
                        TemporalPolicy::pulse(i32::try_from(time).unwrap()),
                    )
                })
                .collect();
            InterventionWorld::new(
                ScientificWorldId::from_raw(u32::try_from(index).unwrap()),
                WorldAction::schedule(
                    Intervention::Sequence(InterventionSequence::new(steps)),
                    TemporalPolicy::sustained(0, 1),
                )
                .unwrap(),
                index == 0,
                WorldBinding::new(
                    TargetPopulation::AllObserved,
                    ObservationUnit::new("independent episode").unwrap(),
                    OutcomeQuantity::mean(
                        VariableId::from_raw(0),
                        QuantityUnits::new("response").unwrap(),
                    )
                    .unwrap(),
                    Conditioning::unconditional(),
                )
                .unwrap(),
                TimeFrame::temporal(
                    2,
                    antecedent_core::HistoryWindow::max_lag(0),
                    antecedent_core::InitialLaw::named("observed baseline history").unwrap(),
                )
                .unwrap(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let source_binding =
        JointDesignSourceBinding::from_world(&worlds[0], design.population.clone()).unwrap();
    let bindings = ReferenceComponentBindings {
        exposure: vec![VariableId::from_raw(1)],
        saturation: vec![],
        mechanism: vec![],
        hybrids: std::array::from_fn(|mask| {
            ScientificWorldId::from_raw(if mask & 1 == 0 { 0 } else { 3 })
        }),
        order: [
            ChangeComponent::Exposure,
            ChangeComponent::Saturation,
            ChangeComponent::Population,
            ChangeComponent::Mechanism,
        ],
    };
    let proof = JointDesignProof {
        version: 1,
        source_binding,
        longitudinal_arms: vec![],
        recipe,
        worlds: worlds
            .iter()
            .enumerate()
            .map(|(index, world)| JointDesignWorldBinding {
                key: index.to_string(),
                world: u32::try_from(index).unwrap(),
                target_identity: joint_design_world_identity(world).to_hex(),
            })
            .collect(),
    };
    let target = ScientificTarget::new(
        TargetForm::Worlds(WorldCollection::new(worlds).unwrap()),
        ModelAssumptions::new(GraphFamily::Unspecified, AssumptionSet::default()),
        InferenceSelection::Bayesian,
    );
    let prepared = PreparedJointDesign::compile(target, proof, &ctx).unwrap();
    let bootstrap =
        JointDesignInference::GaussianLongitudinalBootstrap { replicates: 32, seed: 559 };
    (prepared, bindings, [bootstrap, bayesian])
}

/// Position of each hybrid's world in the design's world axis.
fn indices(prepared: &PreparedJointDesign, bindings: &ReferenceComponentBindings) -> [usize; 16] {
    let worlds = &prepared.proof().worlds;
    std::array::from_fn(|mask| {
        worlds.iter().position(|world| world.world == bindings.hybrids[mask].raw()).unwrap()
    })
}

#[test]
fn bootstrap_design_reference_is_the_hybrid_law_of_its_realizations() {
    let ctx = ExecutionContext::for_tests(559);
    let (prepared, bindings, [bootstrap, _]) = design();
    let result = prepared.execute(bootstrap, &ctx).unwrap();
    let routed = result
        .decompose_reference_components(
            &bindings,
            ScientificConsumptionScope::IdentifiedCausal,
            &ctx,
        )
        .unwrap();
    let indices = indices(&prepared, &bindings);
    let width = prepared.proof().worlds.len();
    let means = &result.law().estimate().means;
    let point = indices.map(|index| means[index]);
    let draws = result
        .law()
        .frequentist_realizations()
        .expect("a bootstrap law keeps its realizations")
        .iter()
        .map(|row| {
            let row = row.as_ref()?;
            (row.len() == width).then(|| indices.map(|index| row[index]))
        })
        .collect::<Vec<_>>();
    let direct =
        engine::decomposition::decompose_hybrid_law(point, &draws, bindings.order, &ctx).unwrap();
    same("bootstrap point", &direct.point, &routed.decomposition.point);
    same("bootstrap draws", &direct.draws, &routed.decomposition.draws);
    same("bootstrap weights", &direct.weights, &routed.decomposition.weights);
}

#[test]
fn posterior_design_reference_covariance_is_the_propagated_hybrid_covariance() {
    let ctx = ExecutionContext::for_tests(559);
    let (prepared, bindings, [_, bayesian]) = design();
    let result = prepared.execute(bayesian, &ctx).unwrap();
    let routed = result
        .decompose_reference_components(
            &bindings,
            ScientificConsumptionScope::IdentifiedCausal,
            &ctx,
        )
        .unwrap();
    let indices = indices(&prepared, &bindings);
    let width = prepared.proof().worlds.len();
    let native = result.law().estimate();
    assert_eq!(native.covariance.len(), width * width, "a posterior law carries its covariance");
    let covariance: [[f64; 16]; 16] = std::array::from_fn(|a| {
        std::array::from_fn(|b| native.covariance[indices[a] * width + indices[b]])
    });
    let direct =
        engine::decomposition::propagate_hybrid_covariance(&covariance, bindings.order).unwrap();
    same("propagated covariance", &Some(direct), &routed.covariance);
}

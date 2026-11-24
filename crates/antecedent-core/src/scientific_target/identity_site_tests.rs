//! Identity sites of the scientific-target encodings (`formal/identity_sites.toml`,
//! `core.scientific_target.*`), checked with the category laws of
//! [`crate::identity_site_laws`] over the fixtures of the scientific-target tests.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::identity_site_laws::{digest, multiset_law, ordered_law, set_law};
use crate::scientific_target::{HistoryIndexedTargetLaw, StructuralAction};

/// Conditioning is a set (sorted and deduplicated by `Conditioning::on`); the world binding
/// writes the stored, sorted list.
#[test]
fn site_conditioning_on_and_world_binding() {
    set_law(&[vid(5), vid(6), vid(7)], |items| {
        Some(
            static_contrast(TargetPopulation::AllObserved, Conditioning::on(items.iter().copied()))
                .identity(),
        )
    });
}

/// The world binding writes the conditioning in its stored order, which `Conditioning::on`
/// makes ascending: a reordering of the declaration never reaches the bytes.
#[test]
fn site_world_binding_conditioning() {
    let declared = [vid(7), vid(5), vid(6)];
    let bound = |items: &[VariableId]| {
        binding(TargetPopulation::AllObserved, Conditioning::on(items.iter().copied()))
    };
    assert_eq!(digest(&bound(&declared)), digest(&bound(&[vid(5), vid(6), vid(7)])));
    assert_eq!(bound(&declared).conditioning().variables(), &[vid(5), vid(6), vid(7)]);
}

fn arms(mediators: &[VariableId]) -> Option<MediationArms> {
    MediationArms::new(
        vid(0),
        mediators.iter().copied(),
        set_level(vid(0), 0.0),
        set_level(vid(0), 1.0),
        MediationComponent::Total,
    )
    .ok()
}

#[test]
fn site_mediation_arms_mediators() {
    set_law(&[vid(5), vid(6), vid(7)], |items| {
        let target = MediationTarget::new(
            binding(TargetPopulation::AllObserved, Conditioning::unconditional()),
            TimeFrame::static_decision(),
            None,
            arms(items)?,
            [],
        )
        .ok()?;
        Some(digest(&target))
    });
}

#[test]
fn site_mediation_target_premises() {
    let premises = [
        CrossWorldPremise::Composition,
        CrossWorldPremise::CrossWorldIndependence,
        CrossWorldPremise::PathSpecificNested,
    ];
    set_law(&premises, |items| {
        let target = MediationTarget::new(
            binding(TargetPopulation::AllObserved, Conditioning::unconditional()),
            TimeFrame::static_decision(),
            None,
            arms(&[vid(2)])?,
            items.iter().copied(),
        )
        .ok()?;
        Some(digest(&target))
    });
}

#[test]
fn site_model_assumptions_records() {
    let records = [
        record(Assumption::CausalMarkov, AssumptionScope::Global, AssumptionStatus::Declared),
        record(Assumption::Positivity, AssumptionScope::Global, AssumptionStatus::Declared),
        record(
            Assumption::Custom { id: Arc::from("x"), description: Arc::from("a") },
            AssumptionScope::Global,
            AssumptionStatus::Declared,
        ),
    ];
    set_law(&records, |items| Some(with_records(items.to_vec()).identity()));
}

/// The scope of an assumption over variables is a set: the encoding sorts and deduplicates.
#[test]
fn site_assumption_scope_variables() {
    set_law(&[vid(5), vid(6), vid(7)], |items| {
        Some(digest(&AssumptionScope::Variables { variables: Arc::from(items) }))
    });
}

/// `WorldCollection::new` sorts worlds by id and refuses a repeated id; the encoding digests
/// each world's content (ids and reference flags dropped) as a multiset.
#[test]
fn site_world_collection() {
    let worlds =
        [static_world(0, true, 0.0), static_world(1, false, 1.0), static_world(2, false, 2.0)];
    set_law(&worlds, |items| WorldCollection::new(items.to_vec()).ok().map(|c| digest(&c)));
    // Two worlds with the same content under different ids are two worlds of the multiset.
    let pair = WorldCollection::new([static_world(0, true, 0.0), static_world(1, false, 1.0)]);
    let triple = WorldCollection::new([
        static_world(0, true, 0.0),
        static_world(1, false, 1.0),
        static_world(2, false, 1.0),
    ]);
    assert_ne!(digest(&pair.unwrap()), digest(&triple.unwrap()));
    multiset_law(&[1.0, 2.0, 3.0], |levels| {
        let worlds: Vec<InterventionWorld> = levels
            .iter()
            .enumerate()
            .map(|(index, &level)| static_world(u32::try_from(index).unwrap(), index == 0, level))
            .collect();
        WorldCollection::new(worlds).ok().map(|c| digest(&c))
    });
}

fn world_of(action: WorldAction) -> Option<[u8; 32]> {
    let world = InterventionWorld::new(
        world_id(0),
        action,
        true,
        binding(TargetPopulation::AllObserved, Conditioning::unconditional()),
        TimeFrame::static_decision(),
    )
    .ok()?;
    WorldCollection::new([world]).ok().map(|c| digest(&c))
}

/// Simultaneous interventions are a set: sorted by target variable, a repeated variable refused.
#[test]
fn site_canonical_interventions() {
    let interventions = [set_level(vid(0), 1.0), set_level(vid(2), 0.0), set_level(vid(3), 1.0)];
    set_law(&interventions, |items| world_of(WorldAction::at_decision(items.to_vec()).ok()?));
}

/// Structural actions are a set: sorted by variable and kind, a conflicting repeat refused.
#[test]
fn site_canonical_structural() {
    let actions = [
        StructuralAction::hard_clamp(vid(0), 1.0).unwrap(),
        StructuralAction::additive_shock(vid(2), 0.5).unwrap(),
        StructuralAction::additive_shock(vid(3), -0.5).unwrap(),
    ];
    set_law(&actions, |items| world_of(WorldAction::structural(items.to_vec()).ok()?));
}

fn linear_contrast(terms: &[(ScientificWorldId, f64)]) -> Option<InterventionContrast> {
    let worlds = WorldCollection::new([
        static_world(0, true, 0.0),
        static_world(1, false, 1.0),
        static_world(2, false, 2.0),
    ])
    .ok()?;
    InterventionContrast::linear(worlds, terms.iter().copied(), world_id(0)).ok()
}

/// Linear terms are a set keyed by world (a repeated world is refused); the encoding writes
/// a per-term digest multiset.
#[test]
fn site_intervention_contrast_linear_terms() {
    let terms = [(world_id(0), -1.0), (world_id(1), 0.5), (world_id(2), 0.5)];
    set_law(&terms, |items| linear_contrast(items).map(|c| digest(&c)));
}

/// Experiments are a set keyed by history (sorted, a repeated history refused); coordinates
/// are ordered (each history's levels align with them).
#[test]
fn site_history_indexed_target_law() {
    let experiments = [(vec![0_u32, 0], 0.25), (vec![0, 1], 0.25), (vec![1, 0], 0.5)];
    set_law(&experiments, |items| {
        HistoryIndexedTargetLaw::new("snapshot", [vid(0), vid(1)], items.to_vec())
            .ok()
            .map(|law| digest(&law))
    });
    ordered_law(&[vid(0), vid(1)], |coordinates| {
        HistoryIndexedTargetLaw::new("snapshot", coordinates.iter().copied(), experiments.to_vec())
            .ok()
            .map(|law| digest(&law))
    });
}

//! Identity sites of antecedent-design (group B) on the production encoders, through the
//! public API: transported selection diagrams, the temporal invariance declaration, the
//! adaptive-trial selection diagram, the bridge selection diagram and the compiled joint
//! experiment's adjustment identity.
//!
//! The law each site obeys comes from its classification in formal/identity_sites.toml
//! (see tests/common/mod.rs).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent_design::adaptive_trial_transport::{TrialMechanism, TrialSelectionDiagram};
use antecedent_design::decision_evidence::{
    AdjustmentKind, ComponentAdjustment, Exposure, JointExperimentSpec, OutcomeComponent,
    compile_joint_experiment,
};
use antecedent_design::temporal_policy_transport::TimeInvariance;
use antecedent_design::transported_bridge::BridgeSelectionDiagram;
use antecedent_design::transported_policy::{MechanismStatus, ModelMechanism, SelectionDiagram};
use proptest::prelude::*;

use common::{adversarial_text, distinct_permuted, injective, multiset_laws, set_laws};

const STATUSES: [MechanismStatus; 2] = [MechanismStatus::Invariant, MechanismStatus::Shifted];

fn model_rows() -> Vec<(ModelMechanism, MechanismStatus)> {
    let mechanisms = [
        ModelMechanism::InitialState,
        ModelMechanism::Transition { action: 0 },
        ModelMechanism::Transition { action: 1 },
        ModelMechanism::Observation { action: 0 },
        ModelMechanism::Utility,
    ];
    mechanisms.iter().flat_map(|&m| STATUSES.iter().map(move |&s| (m, s))).collect()
}

fn trial_rows() -> Vec<(TrialMechanism, MechanismStatus)> {
    [TrialMechanism::StratumLaw, TrialMechanism::OutcomeGivenStratumAndArm]
        .iter()
        .flat_map(|&m| STATUSES.iter().map(move |&s| (m, s)))
        .collect()
}

fn outcomes(names: &[&str]) -> Vec<OutcomeComponent> {
    names
        .iter()
        .map(|name| OutcomeComponent {
            name: (*name).to_owned(),
            variable_id: format!("v_{name}"),
            units: "units".to_owned(),
        })
        .collect()
}

const COMPONENTS: [&str; 4] = ["a", "b", "c", "d"];

fn adjustment(component: &str, coefficient: f64) -> ComponentAdjustment {
    ComponentAdjustment {
        component: component.to_owned(),
        kind: AdjustmentKind::FixedCuped { covariate: format!("pre_{component}"), coefficient },
    }
}

fn joint_spec(adjustments: Vec<ComponentAdjustment>) -> JointExperimentSpec {
    JointExperimentSpec {
        outcomes: outcomes(&COMPONENTS),
        assignment_design: "complete".to_owned(),
        population_id: "population".to_owned(),
        treated_regime_id: "treated".to_owned(),
        control_regime_id: "control".to_owned(),
        horizon: 1,
        adjustments,
        exposure: Exposure::NoBetweenUnitInterference { assumption_id: "sutva".to_owned() },
    }
}

proptest! {
    #![proptest_config(common::config())]

    /// `SelectionDiagram::identity_hex`: mechanism rows are sorted and kept with repeats
    /// (the kernel refuses a repeated mechanism before the diagram is used).
    #[test]
    fn site_transported_policy_selection_diagram(
        (rows, shuffled) in distinct_permuted(prop::sample::select(model_rows()), 0..6),
        source in adversarial_text(),
        target in adversarial_text(),
    ) {
        let identity = |rows: &[(ModelMechanism, MechanismStatus)]| {
            Some(SelectionDiagram {
                source_population: source.clone(),
                target_population: target.clone(),
                mechanisms: rows.to_vec(),
            }.identity_hex())
        };
        multiset_laws(&rows, &shuffled, identity)?;
        let diagram = |parts: &(String, String)| SelectionDiagram {
            source_population: parts.0.clone(),
            target_population: parts.1.clone(),
            mechanisms: rows.clone(),
        };
        injective(
            &(source.clone(), target.clone()),
            &(target.clone(), source.clone()),
            Clone::clone,
            |parts| Some(diagram(parts).identity_hex()),
        )?;
    }

    /// `TimeInvariance::identity_hex`: mechanism rows sorted, repeats kept; periods framed.
    #[test]
    fn site_temporal_time_invariance(
        (rows, shuffled) in distinct_permuted(prop::sample::select(model_rows()), 0..6),
        source in adversarial_text(),
        target in adversarial_text(),
        horizon in 0usize..4,
    ) {
        let declaration = |rows: &[(ModelMechanism, MechanismStatus)], periods: &(String, String)| {
            TimeInvariance {
                source_period: periods.0.clone(),
                target_period: periods.1.clone(),
                horizon,
                mechanisms: rows.to_vec(),
            }
        };
        let periods = (source.clone(), target.clone());
        multiset_laws(&rows, &shuffled, |rows| Some(declaration(rows, &periods).identity_hex()))?;
        injective(
            &periods,
            &(target.clone(), source.clone()),
            Clone::clone,
            |parts| Some(declaration(&rows, parts).identity_hex()),
        )?;
    }

    /// `TrialSelectionDiagram::identity_hex`: mechanism rows sorted, repeats kept.
    #[test]
    fn site_adaptive_trial_selection_diagram(
        (rows, shuffled) in distinct_permuted(prop::sample::select(trial_rows()), 0..4),
        source in adversarial_text(),
        target in adversarial_text(),
    ) {
        let identity = |rows: &[(TrialMechanism, MechanismStatus)]| {
            Some(TrialSelectionDiagram {
                source_population: source.clone(),
                target_population: target.clone(),
                mechanisms: rows.to_vec(),
            }.identity_hex())
        };
        multiset_laws(&rows, &shuffled, identity)?;
    }

    /// `BridgeSelectionDiagram::identity_hex`: selection targets are a set (sort + dedup);
    /// the population names are framed.
    #[test]
    fn site_bridge_selection_diagram(
        (targets, shuffled) in distinct_permuted(adversarial_text(), 0..5),
        source in adversarial_text(),
        target in adversarial_text(),
    ) {
        let identity = |targets: &[String]| {
            Some(BridgeSelectionDiagram {
                source_population: source.clone(),
                target_population: target.clone(),
                selection_targets: targets.to_vec(),
            }.identity_hex())
        };
        set_laws(&targets, &shuffled, identity)?;
        injective(
            &(source.clone(), target.clone()),
            &(target.clone(), source.clone()),
            Clone::clone,
            |parts| Some(BridgeSelectionDiagram {
                source_population: parts.0.clone(),
                target_population: parts.1.clone(),
                selection_targets: targets.clone(),
            }.identity_hex()),
        )?;
    }

    /// `compile_joint_experiment`: adjustments are a set keyed by component (a second
    /// adjustment of one component is refused); the coefficient enters by bits.
    #[test]
    fn site_compile_joint_experiment(
        (components, shuffled) in distinct_permuted(prop::sample::select(COMPONENTS.to_vec()), 0..5),
        coefficient in -2.0f64..2.0,
    ) {
        let identity = |components: &[&str]| {
            let adjustments = components.iter().map(|c| adjustment(c, coefficient)).collect();
            compile_joint_experiment(joint_spec(adjustments)).ok().map(|c| c.adjustment_identity)
        };
        set_laws(&components, &shuffled, identity)?;
        common::bit_exact(|value| {
            compile_joint_experiment(joint_spec(vec![adjustment("a", value)]))
                .ok()
                .map(|c| c.adjustment_identity)
        })?;
    }
}

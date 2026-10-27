// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared parsing/contract declarations for the independent joint-panel workflow example.
//! No fitting or truth-dependent estimator initialization is hidden in this helper.

#![allow(
    dead_code,
    reason = "shared example and integration-test fixture; each consumer uses a subset"
)]

use antecedent::analysis::joint_panel::{
    JointPanelColumns, JointPanelFunctional, JointPanelRequest, JointPanelWorld, joint_panel_target,
};
use antecedent_core::{InferenceSelection, ScientificWorldId, VariableId};
use antecedent_data::{PanelData, PanelUnit, TimeSeriesData};
use antecedent_estimate::joint_panel::{
    PanelChannelHistory, PanelChannelProgram, PanelFitOptions, PanelInference,
    PanelInteractionSpec, PanelOptimizerOptions, PanelProgram, PanelScenario,
};
use std::sync::Arc;

const RANDOMIZED: &str =
    include_str!("../../conformance/response/joint_panel/randomized_interaction.csv");
const WHOLE_PROGRAM: &str =
    include_str!("../../conformance/response/joint_panel/whole_program_interaction.csv");
const CONFOUNDED: &str =
    include_str!("../../conformance/response/joint_panel/confounded_interaction.csv");

pub fn columns() -> JointPanelColumns {
    JointPanelColumns {
        channels: vec![VariableId::from_raw(0), VariableId::from_raw(1)],
        outcome: VariableId::from_raw(2),
        confounder: VariableId::from_raw(3),
        season: VariableId::from_raw(4),
    }
}

fn rows(confounded: bool) -> Vec<Vec<&'static str>> {
    (if confounded { CONFOUNDED } else { RANDOMIZED })
        .lines()
        .skip(1)
        .map(|line| line.split(',').collect())
        .collect()
}

pub fn source(confounded: bool) -> PanelData {
    source_variant(confounded, false, 0.0, false)
}

/// Whole-program experiment from independently retained counterfactual means and AR errors.
pub fn whole_program_source() -> PanelData {
    source_variant(false, true, 0.0, false)
}

#[cfg(test)]
pub fn source_changed_history() -> PanelData {
    source_variant(false, false, 1.0, false)
}

#[cfg(test)]
pub fn source_static() -> PanelData {
    source_variant(false, false, 0.0, true)
}

fn source_variant(
    confounded: bool,
    whole_program: bool,
    history_delta: f64,
    omit_history: bool,
) -> PanelData {
    let rows = if whole_program {
        WHOLE_PROGRAM.lines().skip(1).map(|line| line.split(',').collect()).collect()
    } else {
        rows(confounded)
    };
    let units: Vec<_> = (0..4_u32)
        .map(|group| {
            let group_rows: Vec<_> = rows
                .iter()
                .filter(|row| {
                    row[0].parse::<u32>().unwrap() == group
                        && (!omit_history || row[1].parse::<i32>().unwrap() >= 0)
                })
                .collect();
            let mut source_columns: Vec<Vec<f64>> = [4, 5, 6, 3, 2]
                .into_iter()
                .map(|column| {
                    group_rows
                        .iter()
                        .map(|row| {
                            if row[column].is_empty() {
                                f64::NAN
                            } else {
                                row[column].parse().unwrap()
                            }
                        })
                        .collect()
                })
                .collect();
            if group == 0 {
                source_columns[0][0] += history_delta;
            }
            let series = TimeSeriesData::from_f64_columns(
                ["channel_a", "channel_b", "outcome", "z", "season"]
                    .into_iter()
                    .zip(source_columns.iter().map(Vec::as_slice)),
                604_800_000_000_000,
            )
            .unwrap();
            PanelUnit { unit_id: group, series }
        })
        .collect();
    PanelData::try_new(Arc::from(units)).unwrap()
}

pub fn worlds(confounded: bool) -> Vec<JointPanelWorld> {
    let rows = rows(confounded);
    let scenarios: Vec<_> = (0..4_u32)
        .map(|group| {
            let rows: Vec<_> =
                rows.iter().filter(|row| row[0].parse::<u32>().unwrap() == group).collect();
            PanelScenario {
                group: group.to_string(),
                history: vec![
                    PanelChannelHistory {
                        id: "0".into(),
                        values: [rows[0][4].parse().unwrap(), rows[1][4].parse().unwrap()],
                    },
                    PanelChannelHistory {
                        id: "1".into(),
                        values: [rows[0][5].parse().unwrap(), rows[1][5].parse().unwrap()],
                    },
                ],
                confounder: rows[2..].iter().map(|row| row[3].parse().unwrap()).collect(),
                season: rows[2..].iter().map(|row| row[2].parse().unwrap()).collect(),
            }
        })
        .collect();
    [("reference", 0.0, 0.0), ("a", 4.0, 0.0), ("b", 0.0, 4.0), ("joint", 4.0, 4.0)]
        .into_iter()
        .enumerate()
        .map(|(index, (id, a, b))| JointPanelWorld {
            world: ScientificWorldId::from_raw(u32::try_from(index).unwrap()),
            reference: index == 0,
            program: PanelProgram {
                id: id.into(),
                channels: vec![
                    PanelChannelProgram { id: "0".into(), values: vec![a; 24] },
                    PanelChannelProgram { id: "1".into(), values: vec![b; 24] },
                ],
                scenarios: scenarios.clone(),
            },
        })
        .collect()
}

pub fn request(data: &PanelData, confounded: bool, inference: PanelInference) -> JointPanelRequest {
    let selection = if matches!(&inference, PanelInference::Bayesian { .. }) {
        InferenceSelection::Bayesian
    } else {
        InferenceSelection::Frequentist
    };
    let columns = columns();
    let worlds = worlds(confounded);
    let target = joint_panel_target(
        data.schema(),
        &columns,
        2,
        &worlds,
        JointPanelFunctional::Interaction {
            joint: ScientificWorldId::from_raw(3),
            a_at_reference: ScientificWorldId::from_raw(1),
            b_at_reference: ScientificWorldId::from_raw(2),
            reference: ScientificWorldId::from_raw(0),
        },
        selection,
    )
    .unwrap();
    JointPanelRequest {
        columns,
        interactions: vec![
            PanelInteractionSpec::from_saturated_features(
                "ab",
                &antecedent_core::ValueExpr::var(VariableId::from_raw(0))
                    .mul(antecedent_core::ValueExpr::var(VariableId::from_raw(1))),
            )
            .unwrap(),
        ],
        seasons: 4,
        lag_order: 2,
        options: PanelFitOptions {
            optimizer: PanelOptimizerOptions { max_iterations: 2000, gradient_tolerance: 1e-5 },
            inference,
            initial: None,
        },
        family: antecedent_estimate::joint_panel::PanelFamily::default(),
        worlds,
        target,
        assignment: None,
        temporal: None,
    }
}

/// The fixture's real assignment is per-period and conditional on observed z.
/// Supplying it must not certify an experiment on complete 24-step programs.
pub fn period_randomization() -> antecedent_core::RandomizedEffectQuery {
    let rows = rows(false);
    let rows: Vec<_> = rows.iter().filter(|row| row[1].parse::<i32>().unwrap() >= 0).collect();
    let assignment: Vec<_> = rows
        .iter()
        .map(|row| {
            usize::from(row[4].parse::<f64>().unwrap() > 0.0)
                + 2 * usize::from(row[5].parse::<f64>().unwrap() > 0.0)
        })
        .collect();
    let probabilities: Vec<_> = rows
        .iter()
        .map(|row| {
            let z = row[3].parse::<f64>().unwrap();
            let a = if z < 0.0 { 0.35 } else { 0.65 };
            let b = if z < 0.0 { 0.65 } else { 0.35 };
            vec![(1.0 - a) * (1.0 - b), a * (1.0 - b), (1.0 - a) * b, a * b]
        })
        .collect();
    let labels: Arc<[Arc<str>]> = rows
        .iter()
        .map(|row| Arc::from(format!("group_{}.period_{}", row[0], row[1])))
        .collect::<Vec<_>>()
        .into();
    let realized = assignment.iter().map(|&arm| arm != 0).collect::<Vec<_>>();
    let marginal = assignment
        .iter()
        .zip(&probabilities)
        .map(|(&arm, probabilities)| probabilities[arm])
        .collect::<Vec<_>>();
    let arms: Arc<[Arc<str>]> =
        ["reference", "a", "b", "joint"].into_iter().map(Arc::from).collect::<Vec<_>>().into();
    antecedent_core::RandomizedEffectQuery::with_design(
        antecedent_core::RandomizationDesign::MultiArm {
            assignment: assignment.into(),
            probabilities: probabilities.into(),
            arms,
        },
        VariableId::from_raw(2),
        realized,
        marginal,
        Arc::clone(&labels),
        labels,
        ("reference", "a"),
    )
}

/// Independent four-arm whole-program assignment, one observed independent group per arm.
/// Standard-library independent reference assignment seed194 yields [0,1,2,3];
/// the retained CSV/truth, not production model code, generates these outcomes.
pub fn whole_program_randomization() -> antecedent_core::RandomizedEffectQuery {
    let labels: Arc<[Arc<str>]> =
        (0..4).map(|group| Arc::from(group.to_string())).collect::<Vec<_>>().into();
    let arms: Arc<[Arc<str>]> =
        ["reference", "a", "b", "joint"].into_iter().map(Arc::from).collect::<Vec<_>>().into();
    antecedent_core::RandomizedEffectQuery::with_design(
        antecedent_core::RandomizationDesign::MultiArm {
            assignment: Arc::from([0, 1, 2, 3]),
            probabilities: vec![vec![0.25; 4]; 4].into(),
            arms,
        },
        VariableId::from_raw(2),
        Vec::from([false, true, true, true]),
        vec![0.25; 4],
        Arc::clone(&labels),
        labels,
        ("reference", "a"),
    )
}

/// Explicit Monte Carlo effort for the weak four-group reference; this is not
/// a change to production defaults or to its declared convergence thresholds.
pub fn posterior_sampling() -> antecedent_estimate::joint_panel::PanelSamplingOptions {
    antecedent_estimate::joint_panel::PanelSamplingOptions {
        chains: 4,
        warmup: 8000,
        draws: 12000,
        ..antecedent_estimate::joint_panel::PanelSamplingOptions::default()
    }
}

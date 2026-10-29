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

/// Retained generating truth of the lag-zero (static) panel.
pub struct LagZeroTruth {
    pub intercept: f64,
    pub coefficients: [f64; 2],
    pub interaction: f64,
    pub confounder: f64,
    pub season_effects: [f64; 4],
    pub half_saturation: f64,
    pub innovation_sd: f64,
    pub group_sd: f64,
}

pub fn lag_zero_truth() -> LagZeroTruth {
    LagZeroTruth {
        intercept: 2.0,
        coefficients: [3.0, 2.0],
        interaction: 1.5,
        confounder: 0.7,
        season_effects: [0.0, 0.2, -0.1, 0.1],
        half_saturation: 3.0,
        innovation_sd: 0.2,
        group_sd: 0.35,
    }
}

/// SplitMix64 stream and its Box-Muller normals: a fixed, dependency-free generator so the
/// lag-zero panel is the same panel on every platform and run.
struct FixtureStream(u64);
impl FixtureStream {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// Number of lag-zero groups and follow-up periods per group.
pub const LAG_ZERO_GROUPS: u32 = 12;
pub const LAG_ZERO_PERIODS: usize = 48;

/// Twelve independent groups of 48 periods generated from [`lag_zero_truth`]: no carryover and
/// no serial error, so the rows are `y = c + b_a s(a) + b_b s(b) + b_ab s(a) s(b) + g z +
/// season + u_group + e` with `s(x) = x / (x + k)`. Doses are 0 with probability 0.25 and
/// otherwise uniform on (0, 10); the deterministic stream makes the panel reproducible.
pub fn source_lag_zero() -> PanelData {
    let truth = lag_zero_truth();
    let mut stream = FixtureStream(20_261_029);
    let saturate = |x: f64| x / (x + truth.half_saturation);
    let units: Vec<_> = (0..LAG_ZERO_GROUPS)
        .map(|group| {
            let intercept = truth.group_sd * stream.normal();
            let mut dose = |stream: &mut FixtureStream| {
                if stream.uniform() < 0.25 { 0.0 } else { 10.0 * stream.uniform() }
            };
            let a: Vec<f64> = (0..LAG_ZERO_PERIODS).map(|_| dose(&mut stream)).collect();
            let b: Vec<f64> = (0..LAG_ZERO_PERIODS).map(|_| dose(&mut stream)).collect();
            let z: Vec<f64> = (0..LAG_ZERO_PERIODS).map(|_| stream.normal()).collect();
            let season: Vec<f64> = (0..LAG_ZERO_PERIODS).map(|t| (t % 4) as f64).collect();
            let y: Vec<f64> = (0..LAG_ZERO_PERIODS)
                .map(|t| {
                    let (sa, sb) = (saturate(a[t]), saturate(b[t]));
                    truth.intercept
                        + truth.coefficients[0] * sa
                        + truth.coefficients[1] * sb
                        + truth.interaction * sa * sb
                        + truth.confounder * z[t]
                        + truth.season_effects[t % 4]
                        + intercept
                        + truth.innovation_sd * stream.normal()
                })
                .collect();
            let series = TimeSeriesData::from_f64_columns(
                ["channel_a", "channel_b", "outcome", "z", "season"]
                    .into_iter()
                    .zip([&a, &b, &y, &z, &season].map(|column| column.as_slice())),
                604_800_000_000_000,
            )
            .unwrap();
            PanelUnit { unit_id: group, series }
        })
        .collect();
    PanelData::try_new(Arc::from(units)).unwrap()
}

/// The lag-zero request on [`source_lag_zero`]: the four-world interaction contrast at dose 4
/// held for 24 periods, with every group's own first 24 contexts as the future context.
pub fn request_lag_zero(data: &PanelData, inference: PanelInference) -> JointPanelRequest {
    use antecedent_data::TableView as _;
    let selection = if matches!(&inference, PanelInference::Bayesian { .. }) {
        InferenceSelection::Bayesian
    } else {
        InferenceSelection::Frequentist
    };
    let columns = columns();
    let scenarios: Vec<_> = data
        .units()
        .iter()
        .map(|unit| {
            let z = unit.series.float64_cow(VariableId::from_raw(3)).unwrap();
            let season = unit.series.float64_cow(VariableId::from_raw(4)).unwrap();
            PanelScenario {
                group: unit.unit_id.to_string(),
                history: vec![
                    PanelChannelHistory { id: "0".into(), values: [0.0; 2] },
                    PanelChannelHistory { id: "1".into(), values: [0.0; 2] },
                ],
                confounder: z[..24].to_vec(),
                season: season[..24].iter().map(|value| *value as usize).collect(),
            }
        })
        .collect();
    let worlds: Vec<_> = [("reference", 0.0, 0.0), ("a", 4.0, 0.0), ("b", 0.0, 4.0), ("joint", 4.0, 4.0)]
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
        .collect();
    let target = joint_panel_target(
        data.schema(),
        &columns,
        0,
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
        lag_order: 0,
        options: PanelFitOptions {
            optimizer: PanelOptimizerOptions { max_iterations: 4000, gradient_tolerance: 1e-5 },
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

/// Sampling effort and thresholds for the lag-zero reference: four chains and the strict
/// convergence policy (rank-normalized R-hat at most 1.01, bulk and tail ESS at least 400).
pub fn lag_zero_sampling() -> antecedent_estimate::joint_panel::PanelSamplingOptions {
    antecedent_estimate::joint_panel::PanelSamplingOptions {
        chains: 4,
        warmup: 10_000,
        draws: 25_000,
        max_rhat: 1.01,
        min_ess: 400.0,
        ..antecedent_estimate::joint_panel::PanelSamplingOptions::default()
    }
}

/// Groups and observed periods of the confounded panel with separately moving channels.
pub const CONFOUNDED_VARIED_GROUPS: u32 = 4;
pub const CONFOUNDED_VARIED_PERIODS: usize = 40;

/// A confounded panel whose channels move separately.
///
/// Generating truth (known exactly, by construction): per period the outcome mean is `2 +
/// group + season + 0.7 z + 3 s(a) + 2 s(b) + 1.5 s(a) s(b) + 0.8 u` plus an AR(1) error with
/// coefficient 0.45, where `s(x) = x / (x + 2)` and `a`, `b` are the lag-weighted exposures
/// (weights 0.6/0.3/0.1 and 0.5/0.35/0.15) of the two 0/4 dose channels. The latent `u` is
/// not a column of the table. Both channels are driven by `u`: `channel_a = 4 [u + 0.8 e_a >
/// 0]` and `channel_b = 4 [u + 0.8 e_b > 0]` with independent noises `e_a`, `e_b`. The channels
/// therefore co-move with the outcome's unobserved driver yet are not collinear, so the
/// interaction fit is well-posed and its estimate is biased by `u`: the true interaction
/// contrast is the `1.5` term, and no observational reading of this panel may carry a causal
/// claim. The deterministic stream makes the panel the same on every platform and run.
pub fn source_confounded_varied() -> PanelData {
    let mut stream = FixtureStream(20_261_104);
    let saturate = |x: f64| x / (x + 2.0);
    let units: Vec<_> = (0..CONFOUNDED_VARIED_GROUPS)
        .map(|group| {
            let intercept = 0.35 * stream.normal();
            let n = CONFOUNDED_VARIED_PERIODS;
            let mut a = vec![0.0; 2];
            let mut b = vec![0.0; 2];
            let mut z = vec![f64::NAN; 2];
            let mut u = vec![0.0; 2];
            for _ in 0..n {
                let latent = stream.normal();
                u.push(latent);
                z.push(stream.normal());
                a.push(if latent + 0.8 * stream.normal() > 0.0 { 4.0 } else { 0.0 });
                b.push(if latent + 0.8 * stream.normal() > 0.0 { 4.0 } else { 0.0 });
            }
            let mut y = vec![f64::NAN; 2];
            let mut error = 0.0;
            for t in 0..n {
                let k = t + 2;
                let la = 0.6 * a[k] + 0.3 * a[k - 1] + 0.1 * a[k - 2];
                let lb = 0.5 * b[k] + 0.35 * b[k - 1] + 0.15 * b[k - 2];
                let (sa, sb) = (saturate(la), saturate(lb));
                error = 0.45 * error + 0.2 * stream.normal();
                y.push(
                    2.0 + intercept
                        + [0.0, 0.2, -0.1, 0.1][t % 4]
                        + 0.7 * z[k]
                        + 3.0 * sa
                        + 2.0 * sb
                        + 1.5 * sa * sb
                        + 0.8 * u[k]
                        + error,
                );
            }
            let season: Vec<f64> = (0..n + 2).map(|t| ((t + 2) % 4) as f64).collect();
            let series = TimeSeriesData::from_f64_columns(
                ["channel_a", "channel_b", "outcome", "z", "season"]
                    .into_iter()
                    .zip([&a, &b, &y, &z, &season].map(|column| column.as_slice())),
                604_800_000_000_000,
            )
            .unwrap();
            PanelUnit { unit_id: group, series }
        })
        .collect();
    PanelData::try_new(Arc::from(units)).unwrap()
}

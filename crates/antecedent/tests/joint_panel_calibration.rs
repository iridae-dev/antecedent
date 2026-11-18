//! Coverage of the uncertainty the fitted joint panel publishes.
//!
//! Every test repeats the whole procedure on independent panels simulated from a known
//! generating law: data generation, the compiled panel program, the numerical fit (optimizer,
//! curvature, group-cluster sandwich draws or the Metropolis posterior, or the complete
//! change-selection bootstrap or posterior of the time-varying fit) and the scalar the result
//! publishes for the prespecified contrast, `JointPanelResult::scheduled_contrast` at the
//! terminal period. The interval scored is the one the result returns, at the level it
//! reports. A replicate whose fit executes but publishes no interval (a failed draw slot, an
//! unconverged optimizer or sampler) is scored as a miss; only a fit the library refuses is a
//! skip.
//!
//! Generating law, shared by every panel test. For group `g`, period `t` the outcome is
//!
//! ```text
//! y = c + b0 s0(e0) + b1 s1(e1) + b01 s0(e0) s1(e1) + gamma z + season[t] + u_g + e_gt
//! ```
//!
//! with `e_d` the lag-weighted exposure of channel `d`, `s_d` its saturation (Michaelis-Menten
//! `x / (x + k)` for the reviewed family; Hill `x^h / (x^h + k^h)` with one half-saturation
//! per group class for the broader family), `u_g` a Gaussian group intercept and `e_gt` a
//! stationary Gaussian AR(1) error (no serial error when there is no carryover). This is the
//! mechanism each fit states, so the panels are inside the fitted model.
//!
//! Truth. The target is the interaction contrast of the four worlds (reference, a, b, joint)
//! at the last of 24 periods, each world holding both channels at dose 4 or 0 for the whole
//! horizon. At the last period the exposure window lies wholly inside the program, so the
//! lag weights (which sum to one) return the dose itself, and `s_d(0) = 0`. The four world
//! means share the intercept, the confounder, the season and the group intercepts, which
//! cancel in `joint - a - b + reference`, leaving, for each group, `b01 s0(4) s1(4)`. The
//! contrast averages the groups with equal weight:
//!
//! ```text
//! truth = mean_g  b01  s0(4; class(g))  s1(4; class(g))
//! ```
//!
//! The time-varying panels let `b01` change once, at a declared period inside the candidate
//! set, so the truth at the last period uses the changed value.
//!
//! Frequentist fits are measured at fixed truth (sizes scale with the grid: groups at three
//! quarters, one and two times the base). Bayesian fits are measured by simulation-based
//! calibration instead: the generating law is drawn from the fit's own proper prior
//! (`prior_law` lists every component), a panel is simulated from it, and the published
//! equal-tailed interval is scored against the drawn truth, which a correct posterior covers
//! at exactly its level. That statistic holds at any size, so one small size is measured, with
//! the library's default sampler.
//!
//! Beyond the contrast intervals the file measures what else the panel route publishes:
//! the specification masses of the time-varying fit on panels with no change (a one-sided
//! false-change measurement), the weights of the experiment update applied to the fitted
//! posterior, the Monte Carlo standard error of the value of a future sample, and the
//! contrast and covariance of the prepared Gaussian joint response.
//!
//! The three tests whose replicates cost a complete 100-draw change-selection bootstrap each
//! (the false-change measurement and the two time-varying contrast intervals) are measured by
//! cheaper, validated methods: the contrast intervals by warp-speed pooling of a few inner draws
//! per replicate, the false-change rate by a sequential screen (see `warp_intervals` and
//! `screened_masses_reading`). Both are compared with the nested measurement on the same
//! replicates by the `joint_panel_warp_speed_validation_*` tests, and
//! `ANTECEDENT_CALIBRATION_NESTED=1` runs the nested measurement itself.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "bounded replicate indices and design sizes in a calibration fixture"
)]
#![allow(clippy::cast_precision_loss, reason = "bounded simulation sizes and small indices")]
#![allow(clippy::too_many_lines, reason = "each design keeps its complete setup together")]

mod common;
#[path = "common/warp_speed.rs"]
mod warp_speed;

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent::analysis::joint_panel::{
    ChangeComponent, DecompositionEndpoint, JointPanelColumns, JointPanelFit, JointPanelFunctional,
    JointPanelRequest, JointPanelResponseFunctional, JointPanelResult, JointPanelState,
    JointPanelWorld, MechanismPeriodMap, PanelDecompositionRequest, PanelExperimentDecisionState,
    PanelExperimentEvidence, PanelExperimentMapping, PanelExperimentModularPolicy,
    PanelExperimentUpdate, PanelFutureSample, joint_panel_target,
};
use antecedent::analysis::joint_panel::{
    JointPanelContinuousSearchRequest, JointPanelScientificScope, JointPanelSearchUncertainty,
};
use antecedent::analysis::joint_panel::{JointPanelContrast, JointPanelContrastUncertainty};
use antecedent::{BayesianConfig, InferenceMode, PreparedStudy, RefuteSuite, Study};
use antecedent_contract::bridge::*;
use antecedent_contract::evidence_ledger::*;
use antecedent_core::{
    AssumptionSet, Conditioning, ExecutionContext, GraphFamily, InferenceSelection, Intervention,
    InterventionContrast, InterventionWorld, ModelAssumptions, ObservationUnit, OutcomeQuantity,
    QuantityUnits, ScientificTarget, ScientificWorldId, TargetForm, TargetPopulation, TimeFrame,
    Value, VariableId, WorldAction, WorldBinding, WorldCollection,
};
use antecedent_data::{PanelData, PanelUnit, TableView, TabularData, TimeSeriesData};
use antecedent_design::DesignError;
use antecedent_design::decision::{DecisionProblem, Utility};
use antecedent_design::decision_contract::{
    ProgramObjective, ProgramSearchContract, ProgramSearchSpace, UtilityExpr,
};
use antecedent_design::decision_reliability::{
    SELECTED_VALUE_INTERVAL_METHOD, selected_value_basis,
};
use antecedent_design::program_search::{
    FiniteProgramDomain, ProgramControl, continuous::ContinuousSearchOptions,
};
use antecedent_estimate::joint_panel::time_varying::{
    PanelMeanParameter, TemporalModelChoice, TemporalPanelSpec,
};
use antecedent_estimate::joint_panel::{
    PanelAdstock, PanelChannelHistory, PanelChannelProgram, PanelFamily, PanelFitOptions,
    PanelInference, PanelInteractionSpec, PanelOptimizerOptions, PanelPrior, PanelProgram,
    PanelSamplingOptions, PanelSaturation, PanelSaturationScope, PanelScenario,
};
use antecedent_graph::Dag;
use common::calibration::{
    Construction, CoverageSide, CoverageTally, RecordKey, ScopeFacts, Z95, gaussian, grid_n,
    grid_seed, map_replicates, map_replicates_from, n_sim, normal_interval, quantile_interval,
    replicate_end, replicate_start, stream_seed, uniform,
};
use warp_speed::{Pooling, QuantileRule, Replicate, next_u64, pooled_draw_sets};

const LEVEL: f64 = 0.95;
const CADENCE_NS: u64 = 604_800_000_000_000;
/// Periods of every program and the dose both channels hold in the contrast's non-reference
/// worlds.
const HORIZON: usize = 24;
const DOSE: f64 = 4.0;
/// Draw slots of a frequentist fit (sandwich draws of an invariant fit, whole-procedure
/// bootstrap replicates of a time-varying fit). The library has no default: the caller states
/// the count, and the repository's own workflow uses 100, which is used here. The Monte Carlo
/// noise of a 100-draw interval is part of what is measured.
const FREQUENTIST_DRAWS: usize = 100;
/// Draw slots of the invariant frequentist contrast records. Their draws are Gaussian about the
/// fitted parameters, so the interval is a Monte Carlo percentile of a known law: at 100 draws
/// its finite-draw noise and the interpolated tail ranks alone cost about two points of
/// coverage. The records are measured at this count, which their validation scope states
/// (`replicates_ok`); a run with fewer draws is outside the measured scope.
const INVARIANT_FREQUENTIST_DRAWS: usize = 1000;
/// Groups and periods of the prior-drawn calibration panels: the fewest the model needs (the
/// programs run for 24 periods, a class needs several groups). Calibration under prior-drawn
/// truths is exact at any size, so one size is measured.
const CALIBRATION_GROUPS: usize = 12;
/// Candidate change periods and smoothing grid of the time-varying specification.
const CHANGE_POINTS: [usize; 3] = [8, 12, 16];
const SMOOTH_KNOTS: [usize; 4] = [0, 8, 16, 23];
/// The one change of the time-varying panels: first changed period and the changed
/// interaction coefficient.
const CHANGE_AT: usize = 12;
const CHANGED_INTERACTION: f64 = 2.5;

// ---------------------------------------------------------------------------------------
// The generating law.
// ---------------------------------------------------------------------------------------

/// Which fit a cell measures: carryover and mechanism family.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Cell {
    /// Finite-lag carryover (two earlier periods) with AR(1) errors, or none.
    lagged: bool,
    /// Hill saturation with two group classes (and geometric adstock when lagged), or the
    /// reviewed Michaelis-Menten family.
    broader: bool,
}

impl Cell {
    fn prefix(self) -> usize {
        if self.lagged { 2 } else { 0 }
    }
}

/// What the panel is fitted for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Path {
    /// Invariant mechanism fit on panels with no change.
    Invariant,
    /// Time-varying fit (complete change/smoothing selection) on panels with one change.
    Changed,
    /// Time-varying fit on panels with no change.
    Unchanged,
    /// Time-varying fit on panels whose change (or none) is drawn from the specification's
    /// own prior.
    Prior,
}

impl Path {
    fn varying(self) -> bool {
        self != Self::Invariant
    }
    fn periods(self) -> usize {
        if self.varying() { HORIZON } else { 36 }
    }
    fn temporal(self) -> Option<TemporalPanelSpec> {
        self.varying().then(|| TemporalPanelSpec {
            varying: vec![PanelMeanParameter::Interaction { id: "ab".into() }],
            change_points: CHANGE_POINTS.to_vec(),
            minimum_segment: 4,
            weak_change_gap: 4.0,
            smooth_knots: SMOOTH_KNOTS.to_vec(),
            smoothing: vec![1.0],
            path_prior_sd: 1.0,
        })
    }
    fn groups(self) -> usize {
        if self.varying() { grid_n(24) } else { grid_n(30) }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Frequentist,
    Bayesian,
}

/// Parameters the panels are generated from.
#[derive(Clone)]
struct Law {
    intercept: f64,
    channel: [f64; 2],
    interaction: f64,
    confounder: f64,
    season: [f64; 4],
    /// Half-saturation of class 0 and class 1.
    half_saturation: [f64; 2],
    /// Hill shape per channel; `None` is Michaelis-Menten.
    hill: Option<[f64; 2]>,
    /// Lag weights per channel over the current and two earlier periods.
    lag: [[f64; 3]; 2],
    rho: f64,
    innovation_sd: f64,
    group_sd: f64,
    /// Whether odd groups form a second saturation class.
    classes: bool,
    /// Departure of the interaction coefficient from `interaction` over follow-up periods.
    shift: Shift,
}

/// The path of the interaction coefficient relative to its baseline.
#[derive(Clone, Copy)]
enum Shift {
    None,
    /// Added from this follow-up period on.
    Step {
        at: usize,
        delta: f64,
    },
    /// Piecewise-linear path through zero at period 0 and these values at the smooth knots
    /// after the first.
    Smooth([f64; 3]),
}

/// Normalized truncated geometric weights `(1, theta, theta^2)`.
fn geometric(theta: f64) -> [f64; 3] {
    let sum = 1.0 + theta + theta * theta;
    [1.0 / sum, theta / sum, theta * theta / sum]
}

impl Law {
    fn of(cell: Cell) -> Self {
        let season = [0.0, 0.2, -0.1, 0.1];
        let none = [[1.0, 0.0, 0.0]; 2];
        match (cell.lagged, cell.broader) {
            (false, false) => Self {
                intercept: 2.0,
                channel: [3.0, 2.0],
                interaction: 1.5,
                confounder: 0.7,
                season,
                half_saturation: [3.0, 3.0],
                hill: None,
                lag: none,
                rho: 0.0,
                innovation_sd: 0.2,
                group_sd: 0.35,
                classes: false,
                shift: Shift::None,
            },
            (true, false) => Self {
                intercept: 2.0,
                channel: [3.0, 2.0],
                interaction: 1.5,
                confounder: 0.7,
                season,
                half_saturation: [2.0, 2.0],
                hill: None,
                lag: [[0.6, 0.3, 0.1], [0.5, 0.35, 0.15]],
                rho: 0.45,
                innovation_sd: 0.2,
                group_sd: 0.35,
                classes: false,
                shift: Shift::None,
            },
            (false, true) => Self {
                intercept: 1.0,
                channel: [4.0, 3.0],
                interaction: 1.5,
                confounder: 0.5,
                season,
                half_saturation: [1.0, 3.5],
                hill: Some([2.0, 1.0]),
                lag: none,
                rho: 0.0,
                innovation_sd: 0.2,
                group_sd: 0.35,
                classes: true,
                shift: Shift::None,
            },
            (true, true) => Self {
                intercept: 1.0,
                channel: [4.0, 3.0],
                interaction: 1.5,
                confounder: 0.5,
                season,
                half_saturation: [2.0, 4.5],
                hill: Some([2.0, 1.0]),
                lag: [geometric(0.6), geometric(0.3)],
                rho: 0.3,
                innovation_sd: 0.2,
                group_sd: 0.35,
                classes: true,
                shift: Shift::None,
            },
        }
    }
    fn class_of(&self, group: usize) -> usize {
        if self.classes { group % 2 } else { 0 }
    }
    fn saturate(&self, class: usize, channel: usize, x: f64) -> f64 {
        let k = self.half_saturation[class];
        match self.hill {
            None => x / (x + k),
            Some(shape) => {
                if x <= 0.0 {
                    return 0.0;
                }
                let z = (x / k).powf(shape[channel]);
                z / (1.0 + z)
            }
        }
    }
    /// The interaction coefficient in follow-up period `t`.
    fn interaction_at(&self, t: usize) -> f64 {
        self.interaction
            + match self.shift {
                Shift::None => 0.0,
                Shift::Step { at, delta } => {
                    if t >= at {
                        delta
                    } else {
                        0.0
                    }
                }
                Shift::Smooth(values) => {
                    let heights = [0.0, values[0], values[1], values[2]];
                    let mut at = heights[3];
                    for segment in 0..3 {
                        let (left, right) = (SMOOTH_KNOTS[segment], SMOOTH_KNOTS[segment + 1]);
                        if t <= right {
                            let fraction = (t - left) as f64 / (right - left) as f64;
                            at = heights[segment]
                                + fraction * (heights[segment + 1] - heights[segment]);
                            break;
                        }
                    }
                    at
                }
            }
    }
    /// The analytic terminal interaction contrast over `groups` equally weighted groups.
    fn terminal_contrast(&self, groups: usize) -> f64 {
        let coefficient = self.interaction_at(HORIZON - 1);
        (0..groups)
            .map(|group| {
                let class = self.class_of(group);
                coefficient * self.saturate(class, 0, DOSE) * self.saturate(class, 1, DOSE)
            })
            .sum::<f64>()
            / groups as f64
    }
}

/// One group's simulated series; the first `prefix` rows are pre-period history that carry
/// assignments only.
struct Raw {
    /// The group's realized random intercept.
    intercept: f64,
    a: Vec<f64>,
    b: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    season: Vec<f64>,
}

fn simulate_raw(cell: Cell, law: &Law, class: usize, periods: usize, seed: u64) -> Raw {
    let prefix = cell.prefix();
    let rows = periods + prefix;
    let mut normal = gaussian(stream_seed(seed, 1));
    let mut unit = uniform(stream_seed(seed, 2), 11);
    // Zero with probability 0.2, else uniform on (0, 8): the program dose lies inside.
    let mut dose = || if unit() < 0.2 { 0.0 } else { 8.0 * unit() };
    let a: Vec<f64> = (0..rows).map(|_| dose()).collect();
    let b: Vec<f64> = (0..rows).map(|_| dose()).collect();
    let intercept = law.group_sd * normal();
    let mut error = law.innovation_sd / (1.0 - law.rho * law.rho).sqrt() * normal();
    let mut y = vec![f64::NAN; rows];
    let mut z = vec![f64::NAN; rows];
    let season: Vec<f64> = (0..rows).map(|row| (row % 4) as f64).collect();
    let exposure = |x: &[f64], weights: [f64; 3], row: usize| {
        (0..3).map(|lag| if row >= lag { weights[lag] * x[row - lag] } else { 0.0 }).sum::<f64>()
    };
    for row in prefix..rows {
        let t = row - prefix;
        z[row] = normal();
        let s0 = law.saturate(class, 0, exposure(&a, law.lag[0], row));
        let s1 = law.saturate(class, 1, exposure(&b, law.lag[1], row));
        error = law.rho * error + law.innovation_sd * normal();
        y[row] = law.intercept
            + law.channel[0] * s0
            + law.channel[1] * s1
            + law.interaction_at(t) * s0 * s1
            + law.confounder * z[row]
            + law.season[row % 4]
            + intercept
            + error;
    }
    Raw { intercept, a, b, y, z, season }
}

struct Generated {
    /// Realized random intercept of each group.
    intercepts: Vec<f64>,
    data: PanelData,
    scenarios: Vec<PanelScenario>,
    truth: f64,
    rows: u64,
}

/// Panels with the fixed generating law of the cell (one declared change on `Changed`).
fn generate_panel(cell: Cell, path: Path, groups: usize, seed: u64) -> Generated {
    let mut law = Law::of(cell);
    if path == Path::Changed {
        law.shift = Shift::Step { at: CHANGE_AT, delta: CHANGED_INTERACTION - law.interaction };
    }
    generate_from(cell, &law, path.periods(), groups, seed)
}

/// Panels from an explicit generating law.
fn generate_from(cell: Cell, law: &Law, periods: usize, groups: usize, seed: u64) -> Generated {
    let prefix = cell.prefix();
    let mut units = Vec::with_capacity(groups);
    let mut scenarios = Vec::with_capacity(groups);
    let mut intercepts = Vec::with_capacity(groups);
    for group in 0..groups {
        let raw =
            simulate_raw(cell, law, law.class_of(group), periods, stream_seed(seed, group as u64));
        intercepts.push(raw.intercept);
        let history = |x: &[f64]| if prefix == 2 { [x[0], x[1]] } else { [0.0; 2] };
        scenarios.push(PanelScenario {
            group: group.to_string(),
            history: vec![
                PanelChannelHistory { id: "0".into(), values: history(&raw.a) },
                PanelChannelHistory { id: "1".into(), values: history(&raw.b) },
            ],
            confounder: raw.z[prefix..prefix + HORIZON].to_vec(),
            season: (prefix..prefix + HORIZON).map(|row| row % 4).collect(),
        });
        let series = TimeSeriesData::from_f64_columns(
            ["channel_a", "channel_b", "outcome", "z", "season"]
                .into_iter()
                .zip([&raw.a, &raw.b, &raw.y, &raw.z, &raw.season].map(Vec::as_slice)),
            CADENCE_NS,
        )
        .unwrap();
        units.push(PanelUnit { unit_id: u32::try_from(group).unwrap(), series });
    }
    Generated {
        intercepts,
        data: PanelData::try_new(Arc::from(units)).unwrap(),
        scenarios,
        truth: law.terminal_contrast(groups),
        rows: (groups * periods) as u64,
    }
}

// ---------------------------------------------------------------------------------------
// The request.
// ---------------------------------------------------------------------------------------

fn columns() -> JointPanelColumns {
    JointPanelColumns {
        channels: vec![VariableId::from_raw(0), VariableId::from_raw(1)],
        outcome: VariableId::from_raw(2),
        confounder: VariableId::from_raw(3),
        season: VariableId::from_raw(4),
    }
}

fn worlds(scenarios: &[PanelScenario]) -> Vec<JointPanelWorld> {
    [("reference", 0.0, 0.0), ("a", DOSE, 0.0), ("b", 0.0, DOSE), ("joint", DOSE, DOSE)]
        .into_iter()
        .enumerate()
        .map(|(index, (id, a, b))| JointPanelWorld {
            world: ScientificWorldId::from_raw(u32::try_from(index).unwrap()),
            reference: index == 0,
            program: PanelProgram {
                id: id.into(),
                channels: vec![
                    PanelChannelProgram { id: "0".into(), values: vec![a; HORIZON] },
                    PanelChannelProgram { id: "1".into(), values: vec![b; HORIZON] },
                ],
                scenarios: scenarios.to_vec(),
            },
        })
        .collect()
}

fn family(cell: Cell, groups: usize) -> PanelFamily {
    if !cell.broader {
        return PanelFamily::default();
    }
    PanelFamily {
        adstock: if cell.lagged { PanelAdstock::Geometric } else { PanelAdstock::Simplex },
        saturation: PanelSaturation::Hill,
        scope: PanelSaturationScope::Classes { assignment: (0..groups).map(|g| g % 2).collect() },
        spatial: None,
    }
}

/// The library's default sampler: four chains of 2000 warmup and 2000 retained draws, with
/// the default convergence thresholds (rank-normalized R-hat at most 1.05, bulk and tail ESS
/// at least 100). The fit keeps its own verdict: a posterior that does not meet them
/// publishes no interval.
fn default_sampling() -> PanelSamplingOptions {
    PanelSamplingOptions::default()
}

fn inference(mode: Mode, sampling: PanelSamplingOptions) -> PanelInference {
    match mode {
        Mode::Frequentist => PanelInference::Frequentist { draws: FREQUENTIST_DRAWS },
        Mode::Bayesian => {
            PanelInference::Bayesian { prior: PanelPrior::default(), sampler: sampling }
        }
    }
}

fn request(
    data: &PanelData,
    cell: Cell,
    path: Path,
    groups: usize,
    scenarios: &[PanelScenario],
    inference: PanelInference,
) -> JointPanelRequest {
    let selection = if matches!(&inference, PanelInference::Bayesian { .. }) {
        InferenceSelection::Bayesian
    } else {
        InferenceSelection::Frequentist
    };
    let columns = columns();
    let worlds = worlds(scenarios);
    let target = joint_panel_target(
        data.schema(),
        &columns,
        cell.prefix(),
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
        lag_order: cell.prefix(),
        options: PanelFitOptions {
            optimizer: PanelOptimizerOptions { max_iterations: 4_000, gradient_tolerance: 1e-5 },
            inference,
            initial: None,
        },
        family: family(cell, groups),
        worlds,
        target,
        assignment: None,
        temporal: path.temporal(),
    }
}

type Fitted = (PreparedStudy<JointPanelState>, JointPanelResult, f64, u64, ExecutionContext);

/// Compile and fit one generated panel.
fn fit_generated(
    cell: Cell,
    path: Path,
    mode: Mode,
    sampling: PanelSamplingOptions,
    groups: usize,
    generated: Generated,
    seed: u64,
) -> Result<Fitted, String> {
    fit_generated_with(cell, path, inference(mode, sampling), groups, generated, seed)
}

/// Compile and fit one generated panel under an explicit inference (a frequentist run with a
/// draw count other than [`FREQUENTIST_DRAWS`] is the warp-speed and screening hook).
fn fit_generated_with(
    cell: Cell,
    path: Path,
    inference: PanelInference,
    groups: usize,
    generated: Generated,
    seed: u64,
) -> Result<Fitted, String> {
    let Generated { data, scenarios, truth, rows, .. } = generated;
    let ctx = ExecutionContext::for_tests(grid_seed(seed));
    let request = request(&data, cell, path, groups, &scenarios, inference);
    let prepared = Study::prepare_joint_panel(data, request, &ctx)
        .map_err(|error| format!("prepare: {error}"))?;
    let result = prepared.estimate(&ctx).map_err(|error| format!("estimate: {error}"))?;
    Ok((prepared, result, truth, rows, ctx))
}

/// The compiled and fitted panel of one fixed-truth replicate.
fn fit_panel(
    cell: Cell,
    path: Path,
    mode: Mode,
    groups: usize,
    seed: u64,
) -> Result<Fitted, String> {
    let generated = generate_panel(cell, path, groups, seed);
    fit_generated(cell, path, mode, default_sampling(), groups, generated, seed)
}

/// The compiled and fitted panel of one fixed-truth replicate, frequentist with `draws`
/// whole-procedure slots instead of [`FREQUENTIST_DRAWS`].
fn fit_panel_draws(
    cell: Cell,
    path: Path,
    groups: usize,
    seed: u64,
    draws: usize,
) -> Result<Fitted, String> {
    let generated = generate_panel(cell, path, groups, seed);
    fit_generated_with(cell, path, PanelInference::Frequentist { draws }, groups, generated, seed)
}

// ---------------------------------------------------------------------------------------
// Contrast intervals.
// ---------------------------------------------------------------------------------------

/// One replicate's reading of the published contrast interval.
enum Reading {
    Scored {
        interval: Option<(f64, f64)>,
        truth: f64,
        rows: u64,
        /// Draw slots the fit retained, and how many were complete.
        slots: u32,
        complete: u32,
        /// For a time-varying posterior under prior-drawn truth: whether the smallest set of
        /// specifications holding the level of the published mass contains the true one.
        choice_covered: Option<bool>,
    },
    /// The library refused the fit.
    Refused(String),
}

fn contrast_reading(cell: Cell, path: Path, mode: Mode, groups: usize, seed: u64) -> Reading {
    let fitted = if mode == Mode::Frequentist && !path.varying() {
        fit_panel_draws(cell, path, groups, seed, INVARIANT_FREQUENTIST_DRAWS)
    } else {
        fit_panel(cell, path, mode, groups, seed)
    };
    let (_, result, truth, rows, _) = match fitted {
        Ok(fitted) => fitted,
        Err(reason) => return Reading::Refused(reason),
    };
    let contrast =
        match result.scheduled_contrast(&JointPanelResponseFunctional::TerminalMean, LEVEL) {
            Ok(contrast) => contrast,
            Err(error) => return Reading::Refused(format!("contrast: {error}")),
        };
    assert!((contrast.level - LEVEL).abs() < 1e-12);
    Reading::Scored {
        interval: contrast.interval,
        truth,
        rows,
        slots: u32::try_from(contrast.draws.len()).unwrap(),
        complete: u32::try_from(contrast.draws.iter().flatten().count()).unwrap(),
        choice_covered: None,
    }
}

fn interval_label(mode: Mode, path: Path) -> &'static str {
    match (mode, path.varying()) {
        (Mode::Bayesian, _) => "posterior_quantile",
        (Mode::Frequentist, false) => "sandwich_draw_quantile",
        (Mode::Frequentist, true) => "percentile_bootstrap",
    }
}

/// The construction a record describes. The panel route reports no runtime calibration key,
/// so it is declared here, as for the other routes without one.
fn construction(cell: Cell, path: Path, mode: Mode, calibration: bool) -> Construction {
    let estimator = format!(
        "joint_panel.{}.{}.{}",
        if cell.lagged { "finite_lag" } else { "static" },
        if cell.broader { "hill_classes" } else { "michaelis_menten" },
        if path.varying() { "time_varying_selection" } else { "invariant" },
    );
    Construction {
        query: "JointPanel".into(),
        graph_class: "NotApplicable".into(),
        structure: "panel".into(),
        modality: if calibration {
            "scheduled_contrast_posterior_calibration"
        } else {
            "scheduled_contrast"
        }
        .into(),
        inference: match mode {
            Mode::Frequentist => "Frequentist",
            Mode::Bayesian => "Bayesian",
        }
        .into(),
        estimator,
        interval_method: interval_label(mode, path).into(),
        se_kind: String::new(),
        dependence: if cell.lagged {
            "independent_groups_ar1_random_intercept"
        } else {
            "independent_groups_random_intercept"
        }
        .into(),
        posterior: match mode {
            Mode::Frequentist => "",
            Mode::Bayesian => "proper_prior_metropolis",
        }
        .into(),
        functional: "terminal_interaction_contrast".into(),
        identification: "model_conditional".into(),
        reported_level: LEVEL,
    }
}

fn measure_contrast(test: &'static str, cell: Cell, path: Path, mode: Mode, seed_base: u64) {
    // The time-varying frequentist interval is measured by the warp-speed method (see below);
    // `ANTECEDENT_CALIBRATION_NESTED=1` runs the full nested measurement instead.
    if path.varying() && mode == Mode::Frequentist && !nested_requested() {
        measure_contrast_warp(test, cell, path, seed_base);
        return;
    }
    let groups = path.groups();
    let readings =
        map_replicates(n_sim(), |rep| contrast_reading(cell, path, mode, groups, seed_base + rep));
    score_contrast(test, "generate_panel", cell, path, mode, false, &readings);
}

/// Fold readings into one record and hold it to the gate.
fn score_contrast(
    test: &'static str,
    dgp: &'static str,
    cell: Cell,
    path: Path,
    mode: Mode,
    calibration: bool,
    readings: &[Reading],
) {
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp, interval: interval_label(mode, path) },
        LEVEL,
    );
    let mut refusals = BTreeMap::<String, usize>::new();
    let mut withheld = 0_usize;
    for reading in readings {
        match reading {
            Reading::Refused(reason) => {
                tally.skip();
                *refusals.entry(reason.clone()).or_default() += 1;
            }
            Reading::Scored { interval, truth, rows, slots, complete, .. } => {
                withheld += usize::from(interval.is_none());
                tally.bind_validation(
                    &construction(cell, path, mode, calibration),
                    ScopeFacts {
                        row_count: *rows,
                        replicates_ok: (mode == Mode::Frequentist).then_some(*complete),
                        posterior_draws: (mode == Mode::Bayesian).then_some(*slots),
                        unidentified_mass: 0.0,
                    },
                    "not_applicable",
                );
                tally.record(*interval, *truth);
            }
        }
    }
    eprintln!("{test}: {withheld} fit(s) executed and published no interval (scored as misses)");
    for (reason, count) in &refusals {
        eprintln!("{test}: {count} replicate(s) refused: {reason}");
    }
    tally.assert();
}

// ---------------------------------------------------------------------------------------
// Warp-speed measurement of the time-varying contrast interval.
// ---------------------------------------------------------------------------------------

/// Inner whole-procedure draws each outer replicate runs under the warp-speed measurement.
const WARP_DRAWS: usize = 5;
/// The data-generation label of a warp-speed record: the panels are those of
/// `generate_panel`, the interval is composed from the pooled draws of all replicates.
const WARP_DGP: &str = "generate_panel_warp_speed";
/// Replicates of a warp-speed validation run, nested and warp-speed on the same seeds.
const VALIDATION_REPLICATES: u32 = 100;

/// Whether the full nested measurement is requested for a test that is warp-speed by default.
fn nested_requested() -> bool {
    std::env::var("ANTECEDENT_CALIBRATION_NESTED").is_ok_and(|value| value == "1")
}

/// One replicate of the warp-speed measurement: the point contrast and the few inner draws.
enum WarpReading {
    Scored {
        point: f64,
        /// Inner draws of the contrast; a failed or nonfinite slot stays `None`.
        draws: Vec<Option<f64>>,
        truth: f64,
        rows: u64,
    },
    /// The library refused the fit; a refused replicate is scored as a miss.
    Refused,
}

/// The fit of one replicate with `draws` inner whole-procedure draws: the product's own fit and
/// contrast assembly, with the draw count the only difference from the nested measurement. The
/// interval the product forms from so few draws is not read.
fn warp_reading(cell: Cell, path: Path, groups: usize, seed: u64, draws: usize) -> WarpReading {
    let Ok((_, result, truth, rows, _)) = fit_panel_draws(cell, path, groups, seed, draws) else {
        return WarpReading::Refused;
    };
    match result.scheduled_contrast(&JointPanelResponseFunctional::TerminalMean, LEVEL) {
        Ok(contrast) => {
            WarpReading::Scored { point: contrast.point, draws: contrast.draws, truth, rows }
        }
        Err(_) => WarpReading::Refused,
    }
}

/// Every replicate's interval, formed as the product forms it from the draws of its own law.
///
/// Why this measures the published interval. The time-varying interval is the interpolated
/// equal-tailed percentile of the `FREQUENTIST_DRAWS` whole-procedure contrast draws
/// (`JointPanelContrast::uncertainty_from_draws`): `point + q(draw - point)`. Each draw is the
/// contrast of one parametric bootstrap panel, drawn from the fitted model of the replicate and
/// passed through the complete change-selection procedure, and the draws of one replicate are
/// independent given its data. Warp-speed Monte Carlo (Giacomini, Politis and White 2013)
/// replaces the conditional law of the root `draw - point` given one replicate's data by the
/// pooled law of the roots of every replicate: a replicate then needs a few draws, not a hundred.
/// The pooled roots are the unconditional mixture of the replicates' bootstrap laws.
///
/// What is reproduced and what is not.
///
/// * First order only. The product's interval uses the law conditional on the replicate; the
///   pooled law agrees with it when the conditional law of the root converges in probability to
///   a constant law, and differs at the order of the data dependence of that law (its scale,
///   its skewness, the mixture over selected specifications) otherwise. The roots are pooled
///   unscaled: no standardizing quantity exists that is independent of the draws, and one built
///   from five draws would put its own noise into every interval. The validation test compares
///   the result with the nested measurement.
/// * Finite-draw noise is kept. The product's interval rests on `FREQUENTIST_DRAWS` draws, whose
///   order statistics are noisy; pooling every draw would measure the coverage of an interval
///   with unboundedly many. Each replicate therefore composes a set of `FREQUENTIST_DRAWS`
///   roots chosen at random without replacement from the pool.
/// * The product withholds the interval when any of its slots fails. The pooled slots carry the
///   same failure rate, and a replicate is withheld with the probability that one of its
///   `FREQUENTIST_DRAWS` slots fails at that rate.
fn warp_intervals(readings: &[WarpReading], seed: u64) -> Vec<Option<(f64, f64)>> {
    let replicates: Vec<Option<Replicate>> = readings
        .iter()
        .map(|reading| match reading {
            WarpReading::Scored { point, draws, .. } => {
                Some(Replicate::with_draws(*point, draws.iter().flatten().copied().collect()))
            }
            WarpReading::Refused => None,
        })
        .collect();
    let (mut slots, mut failed) = (0_usize, 0_usize);
    for reading in readings {
        if let WarpReading::Scored { draws, .. } = reading {
            slots += draws.len();
            failed += draws.iter().filter(|draw| draw.is_none()).count();
        }
    }
    let failure_rate = failed as f64 / slots.max(1) as f64;
    let withheld = 1.0 - (1.0 - failure_rate).powi(i32::try_from(FREQUENTIST_DRAWS).unwrap());
    let pooling = Pooling { level: LEVEL, min_pool: 2, inner: Some(FREQUENTIST_DRAWS), seed };
    let sets = pooled_draw_sets(&replicates, QuantileRule::Interpolated, pooling);
    sets.into_iter()
        .enumerate()
        .map(|(index, set)| {
            let set = set?;
            let mut state = seed ^ (index as u64).wrapping_mul(0xA24B_AED4_963E_E407);
            let unit = (next_u64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
            if unit < withheld {
                return None;
            }
            let draws: Vec<Option<f64>> = set.into_iter().map(Some).collect();
            JointPanelContrast::uncertainty_from_draws(
                &draws,
                LEVEL,
                JointPanelContrastUncertainty::EstimatorSampling,
                true,
            )
            .1
        })
        .collect()
}

/// A warp-speed record pools draws across replicates, so it needs the whole replicate range in
/// one process; replicate shards and extensions would pool within a shard only.
fn require_unsharded(test: &str) {
    assert!(
        replicate_start() == 0 && replicate_end().is_none(),
        "{test}: the warp-speed measurement pools draws across all replicates and cannot be \
         sharded or extended; run it whole"
    );
}

/// Warp-speed measurement of the published time-varying contrast interval: `WARP_DRAWS` inner
/// draws per outer replicate, intervals composed from the pooled draws (`warp_intervals`), and
/// the record marked by the `WARP_DGP` data-generation label.
fn measure_contrast_warp(test: &'static str, cell: Cell, path: Path, seed_base: u64) {
    require_unsharded(test);
    let groups = path.groups();
    let readings = map_replicates(n_sim(), |rep| {
        warp_reading(cell, path, groups, seed_base + rep, WARP_DRAWS)
    });
    score_warp_contrast(test, cell, path, &readings, seed_base);
}

fn score_warp_contrast(
    test: &'static str,
    cell: Cell,
    path: Path,
    readings: &[WarpReading],
    seed: u64,
) {
    let mode = Mode::Frequentist;
    let intervals = warp_intervals(readings, seed);
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp: WARP_DGP, interval: interval_label(mode, path) },
        LEVEL,
    );
    let (mut refused, mut withheld) = (0_usize, 0_usize);
    let draws = u32::try_from(FREQUENTIST_DRAWS).unwrap();
    for (reading, interval) in readings.iter().zip(&intervals) {
        match reading {
            WarpReading::Refused => {
                tally.skip();
                refused += 1;
            }
            WarpReading::Scored { truth, rows, .. } => {
                withheld += usize::from(interval.is_none());
                // The scope is that of the interval the product forms: all of its draw slots
                // when it publishes one, at least one failed slot when it withholds it.
                tally.bind_validation(
                    &construction(cell, path, mode, false),
                    ScopeFacts {
                        row_count: *rows,
                        replicates_ok: Some(if interval.is_some() { draws } else { draws - 1 }),
                        posterior_draws: None,
                        unidentified_mass: 0.0,
                    },
                    "not_applicable",
                );
                tally.record(*interval, *truth);
            }
        }
    }
    eprintln!(
        "{test}: warp-speed measurement, {WARP_DRAWS} inner draws per replicate pooled over \
         {} replicate(s); {withheld} composed interval(s) withheld by a failed slot (scored as \
         misses), {refused} refused",
        readings.len()
    );
    tally.assert();
}

/// Coverage of both measurements on the same replicates: the nested interval of the product
/// and the warp-speed interval composed from `WARP_DRAWS` inner draws of the same fits.
fn validate_warp_contrast(cell: Cell, seed_base: u64) {
    let (path, mode) = (Path::Changed, Mode::Frequentist);
    let groups = path.groups();
    let nested = map_replicates_from(0, VALIDATION_REPLICATES, |rep| {
        contrast_reading(cell, path, mode, groups, seed_base + rep)
    });
    let warp = map_replicates_from(0, VALIDATION_REPLICATES, |rep| {
        warp_reading(cell, path, groups, seed_base + rep, WARP_DRAWS)
    });
    let composed = warp_intervals(&warp, seed_base);
    let covers = |interval: &Option<(f64, f64)>, truth: f64| {
        interval.is_some_and(|(low, high)| low <= truth && truth <= high)
    };
    let (mut pairs, mut nested_hits, mut warp_hits, mut disagree) = (0_u32, 0_u32, 0_u32, 0_u32);
    let (mut nested_width, mut warp_width) = (0.0, 0.0);
    for ((nested, warp), composed) in nested.iter().zip(&warp).zip(&composed) {
        let (Reading::Scored { interval, truth, .. }, WarpReading::Scored { .. }) = (nested, warp)
        else {
            continue;
        };
        pairs += 1;
        let (a, b) = (covers(interval, *truth), covers(composed, *truth));
        nested_hits += u32::from(a);
        warp_hits += u32::from(b);
        disagree += u32::from(a != b);
        nested_width += interval.map_or(0.0, |(low, high)| high - low);
        warp_width += composed.map_or(0.0, |(low, high)| high - low);
    }
    let n = f64::from(pairs.max(1));
    let (nested_rate, warp_rate) = (f64::from(nested_hits) / n, f64::from(warp_hits) / n);
    let mcse = (nested_rate * (1.0 - nested_rate) / n + warp_rate * (1.0 - warp_rate) / n).sqrt();
    eprintln!(
        "warp-speed validation over {pairs} paired replicate(s): nested coverage {nested_rate:.4}, \
         warp-speed coverage {warp_rate:.4}, combined MCSE {mcse:.4}, coverage verdicts differ in \
         {disagree}; mean width nested {:.4}, warp-speed {:.4}",
        nested_width / n,
        warp_width / n
    );
    assert!(
        (nested_rate - warp_rate).abs() <= 2.0 * mcse.max(1.0 / n),
        "the warp-speed coverage {warp_rate:.4} differs from the nested coverage {nested_rate:.4} \
         by more than two combined MCSE ({mcse:.4})"
    );
}

#[test]
#[ignore = "warp-speed validation: runs the nested measurement on 100 replicates; run by hand"]
fn joint_panel_warp_speed_validation_time_varying_contrast() {
    validate_warp_contrast(LAGGED, 32_000_000);
}

#[test]
#[ignore = "warp-speed validation: runs the nested measurement on 100 replicates; run by hand"]
fn joint_panel_warp_speed_validation_time_varying_contrast_broader() {
    validate_warp_contrast(BROADER_LAGGED, 32_200_000);
}

// ---------------------------------------------------------------------------------------
// Posterior calibration under prior-drawn truths.
// ---------------------------------------------------------------------------------------

/// Draw the generating law from the prior the Bayesian fit declares (`PanelPrior::default()`
/// and the temporal specification's path prior), so that a correct posterior is calibrated
/// exactly. Every component is a proper density: coefficients, intercept, confounder and
/// non-reference seasons are Normal(0, 10^2); `log k` is Normal(0, 1); each lag simplex is
/// Dirichlet(1, 1, 1) (the geometric decay is Beta(1, 1), uniform); the Hill log shapes are
/// Normal(0, 0.5^2) and a second class's `log k` is Normal(`log k` of class zero, 0.5^2);
/// `rho` is uniform on (-1, 1); the innovation and group standard deviations are half-Normal
/// with scale 2. For the time-varying fit the specification is uniform over no change, each
/// change period and each smoothing penalty, with a change a Normal(0, `path_prior_sd^2`)
/// offset and a smooth path a Gaussian random walk over the knots whose step variance is
/// `path_prior_sd^2 / penalty`. No prior component is improper or data dependent.
fn prior_law(cell: Cell, varying: bool, seed: u64) -> (Law, Option<TemporalModelChoice>) {
    let prior = PanelPrior::default();
    assert!(prior.lag.is_empty(), "the drawn simplexes are Dirichlet(1, 1, 1)");
    assert_eq!(
        prior.family.geometric_decay.map(f64::to_bits),
        [1.0_f64, 1.0].map(f64::to_bits),
        "the drawn decay is Beta(1, 1)"
    );
    let mut normal = gaussian(stream_seed(seed, 901));
    let mut unit = uniform(stream_seed(seed, 902), 17);
    let c = prior.coefficient_sd;
    let intercept = c * normal();
    let channel = [c * normal(), c * normal()];
    let interaction = c * normal();
    let confounder = c * normal();
    let season = [0.0, c * normal(), c * normal(), c * normal()];
    let k0 = (prior.log_half_saturation_mean + prior.log_half_saturation_sd * normal()).exp();
    let family = &prior.family;
    let (half_saturation, hill) = if cell.broader {
        let k1 = (k0.ln() + family.class_log_saturation_sd * normal()).exp();
        let shape = [
            (family.log_hill_shape_mean + family.log_hill_shape_sd * normal()).exp(),
            (family.log_hill_shape_mean + family.log_hill_shape_sd * normal()).exp(),
        ];
        ([k0, k1], Some(shape))
    } else {
        ([k0, k0], None)
    };
    let lag = if !cell.lagged {
        [[1.0, 0.0, 0.0]; 2]
    } else if cell.broader {
        [geometric(unit()), geometric(unit())]
    } else {
        let mut dirichlet = || {
            let e = [-unit().ln(), -unit().ln(), -unit().ln()];
            let sum = e[0] + e[1] + e[2];
            [e[0] / sum, e[1] / sum, e[2] / sum]
        };
        [dirichlet(), dirichlet()]
    };
    let rho = if cell.lagged { 2.0 * unit() - 1.0 } else { 0.0 };
    let innovation_sd = prior.innovation_scale * normal().abs();
    let group_sd = prior.group_scale * normal().abs();
    let (shift, choice) = if varying {
        let spec = Path::Prior.temporal().expect("a time-varying specification");
        let candidates = 1 + spec.change_points.len() + spec.smoothing.len();
        let index = ((unit() * candidates as f64) as usize).min(candidates - 1);
        if index == 0 {
            (Shift::None, TemporalModelChoice::NoChange)
        } else if index <= spec.change_points.len() {
            let at = spec.change_points[index - 1];
            (
                Shift::Step { at, delta: spec.path_prior_sd * normal() },
                TemporalModelChoice::Piecewise { at },
            )
        } else {
            let penalty = spec.smoothing[index - 1 - spec.change_points.len()];
            let step = spec.path_prior_sd / penalty.sqrt();
            let mut values = [0.0; 3];
            let mut level = 0.0;
            for value in &mut values {
                level += step * normal();
                *value = level;
            }
            (Shift::Smooth(values), TemporalModelChoice::Smooth { penalty })
        }
    } else {
        (Shift::None, TemporalModelChoice::NoChange)
    };
    let law = Law {
        intercept,
        channel,
        interaction,
        confounder,
        season,
        half_saturation,
        hill,
        lag,
        rho,
        innovation_sd,
        group_sd,
        classes: cell.broader,
        shift,
    };
    (law, varying.then_some(choice))
}

fn calibration_reading(cell: Cell, varying: bool, seed: u64) -> Reading {
    let path = if varying { Path::Prior } else { Path::Invariant };
    let (law, true_choice) = prior_law(cell, varying, seed);
    let generated = generate_from(cell, &law, HORIZON, CALIBRATION_GROUPS, seed);
    let (_, result, truth, rows, _) = match fit_generated(
        cell,
        path,
        Mode::Bayesian,
        default_sampling(),
        CALIBRATION_GROUPS,
        generated,
        seed,
    ) {
        Ok(fitted) => fitted,
        Err(reason) => return Reading::Refused(reason),
    };
    let contrast =
        match result.scheduled_contrast(&JointPanelResponseFunctional::TerminalMean, LEVEL) {
            Ok(contrast) => contrast,
            Err(error) => return Reading::Refused(format!("contrast: {error}")),
        };
    // The smallest set of specifications, taken by decreasing published mass, that holds the
    // level; a posterior that failed its diagnostics publishes none and scores a miss.
    let choice_covered = true_choice.map(|truth| match &result.fit {
        JointPanelFit::Varying(fit) if fit.posterior_converged == Some(true) => {
            let total: usize = fit.selection_counts.iter().map(|(_, count)| *count).sum();
            let mut order: Vec<&(TemporalModelChoice, usize)> =
                fit.selection_counts.iter().collect();
            order.sort_by_key(|entry| std::cmp::Reverse(entry.1));
            let (mut cumulative, mut hit) = (0_usize, false);
            for entry in order {
                cumulative += entry.1;
                hit |= entry.0 == truth;
                if cumulative as f64 >= LEVEL * total as f64 {
                    break;
                }
            }
            total > 0 && hit
        }
        _ => false,
    });
    Reading::Scored {
        interval: contrast.interval,
        truth,
        rows,
        slots: u32::try_from(contrast.draws.len()).unwrap(),
        complete: u32::try_from(contrast.draws.iter().flatten().count()).unwrap(),
        choice_covered,
    }
}

/// Simulation-based calibration of one Bayesian fit: truths drawn from the fit's own prior, a
/// panel simulated from each, the published equal-tailed interval of the terminal contrast at
/// the published level scored against the drawn truth. Under prior-drawn truths a correct
/// posterior covers at exactly its level, so the two-sided gate is the calibration statistic.
/// The time-varying fits also score the specification: the smallest set of specifications
/// holding the level of the published mass must contain the drawn one (an at-least claim,
/// exact for probabilities and conservative only by the discreteness of the set).
fn measure_calibration(test: &'static str, cell: Cell, varying: bool, seed_base: u64) {
    let path = if varying { Path::Prior } else { Path::Invariant };
    let readings =
        map_replicates(n_sim(), |rep| calibration_reading(cell, varying, seed_base + rep));
    if varying {
        let mut set = CoverageTally::new(format!("{test} specification set"), LEVEL);
        for reading in &readings {
            match reading {
                Reading::Scored { choice_covered, .. } => {
                    let hit = *choice_covered == Some(true);
                    set.record(Some((0.0, 1.0)), if hit { 0.5 } else { 2.0 });
                }
                Reading::Refused(_) => set.skip(),
            }
        }
        score_contrast(test, "generate_from", cell, path, Mode::Bayesian, true, &readings);
        set.assert_at_least();
    } else {
        score_contrast(test, "generate_from", cell, path, Mode::Bayesian, true, &readings);
    }
}

macro_rules! calibration_test {
    ($name:ident, $cell:expr, $varying:expr, $seed:expr) => {
        #[test]
        #[ignore = "coverage: measure with scripts/measure_calibration.sh"]
        fn $name() {
            measure_calibration(stringify!($name), $cell, $varying, $seed);
        }
    };
}

calibration_test!(joint_panel_posterior_calibration_static_l95, STATIC, false, 31_100_000);
calibration_test!(joint_panel_posterior_calibration_lagged_l95, LAGGED, false, 31_300_000);
calibration_test!(
    joint_panel_posterior_calibration_broader_static_l95,
    BROADER_STATIC,
    false,
    31_500_000
);
calibration_test!(
    joint_panel_posterior_calibration_broader_lagged_l95,
    BROADER_LAGGED,
    false,
    31_700_000
);
calibration_test!(joint_panel_posterior_calibration_time_varying_l95, LAGGED, true, 32_100_000);
calibration_test!(
    joint_panel_posterior_calibration_time_varying_broader_l95,
    BROADER_LAGGED,
    true,
    32_300_000
);

const STATIC: Cell = Cell { lagged: false, broader: false };
const LAGGED: Cell = Cell { lagged: true, broader: false };
const BROADER_STATIC: Cell = Cell { lagged: false, broader: true };
const BROADER_LAGGED: Cell = Cell { lagged: true, broader: true };

macro_rules! contrast_test {
    ($name:ident, $cell:expr, $path:expr, $mode:expr, $seed:expr) => {
        #[test]
        #[ignore = "coverage: measure with scripts/measure_calibration.sh"]
        fn $name() {
            measure_contrast(stringify!($name), $cell, $path, $mode, $seed);
        }
    };
}

contrast_test!(
    joint_panel_contrast_static_frequentist_l95,
    STATIC,
    Path::Invariant,
    Mode::Frequentist,
    31_000_000
);
contrast_test!(
    joint_panel_contrast_lagged_frequentist_l95,
    LAGGED,
    Path::Invariant,
    Mode::Frequentist,
    31_200_000
);
contrast_test!(
    joint_panel_contrast_broader_static_frequentist_l95,
    BROADER_STATIC,
    Path::Invariant,
    Mode::Frequentist,
    31_400_000
);
contrast_test!(
    joint_panel_contrast_broader_lagged_frequentist_l95,
    BROADER_LAGGED,
    Path::Invariant,
    Mode::Frequentist,
    31_600_000
);

// The time-varying fit: one change at a declared candidate period; the interval carries the
// complete selection among no change, three change periods and a smooth path.
contrast_test!(
    joint_panel_time_varying_contrast_frequentist_l95,
    LAGGED,
    Path::Changed,
    Mode::Frequentist,
    32_000_000
);
contrast_test!(
    joint_panel_time_varying_contrast_broader_frequentist_l95,
    BROADER_LAGGED,
    Path::Changed,
    Mode::Frequentist,
    32_200_000
);

// ---------------------------------------------------------------------------------------
// Specification masses on panels with no change.
// ---------------------------------------------------------------------------------------

/// What one no-change replicate published about the selected specification.
enum Masses {
    Published {
        /// Mass the law puts on every specification with a change (a change period or a
        /// smooth path).
        change: f64,
        /// Whether the point fit itself selected a change.
        point_change: bool,
        /// Whether every draw slot of the fit completed.
        all_draws: bool,
    },
    /// The first draws of the replicate already exclude a false confident change, so the
    /// remaining draws were not run (see `screened_masses_reading`).
    Screened {
        /// Whether the point fit itself selected a change.
        point_change: bool,
    },
    /// The fit executed and published no masses.
    Absent(String),
    Refused(String),
}

/// The masses of one replicate from `draws` whole-procedure draws ([`FREQUENTIST_DRAWS`] is
/// the published count).
fn masses_reading(cell: Cell, mode: Mode, groups: usize, seed: u64, draws: usize) -> Masses {
    let fitted = match mode {
        Mode::Frequentist => fit_panel_draws(cell, Path::Unchanged, groups, seed, draws),
        Mode::Bayesian => fit_panel(cell, Path::Unchanged, mode, groups, seed),
    };
    let (_, result, _, _, ctx) = match fitted {
        Ok(fitted) => fitted,
        Err(reason) => return Masses::Refused(reason),
    };
    // Forward decomposition between the early and late fitted mechanism periods: the
    // selection masses are carried through the decomposed law.
    let worlds = &result.request().worlds;
    let horizon = worlds[0].horizon();
    let n_groups = result.prepared().model_data().groups.len();
    let weights = vec![1.0 / n_groups as f64; n_groups];
    let endpoint = |index: usize| DecompositionEndpoint {
        program: worlds[index].program.clone(),
        population_weights: weights.clone(),
    };
    let mut terminal = vec![0.0; horizon];
    terminal[horizon - 1] = 1.0;
    let decomposition_request = PanelDecompositionRequest {
        before: endpoint(0),
        after: endpoint(3),
        time_weights: terminal,
        order: [
            ChangeComponent::Exposure,
            ChangeComponent::Saturation,
            ChangeComponent::Population,
            ChangeComponent::Mechanism,
        ],
    };
    let periods =
        MechanismPeriodMap { before: vec![0; horizon], after: vec![horizon - 1; horizon] };
    let decomposition = match result.decompose_reference(&decomposition_request, &periods, &ctx) {
        Ok(decomposition) => decomposition,
        Err(error) => return Masses::Absent(format!("decomposition: {error}")),
    };
    let split = match result.decomposition_selection(&decomposition) {
        Ok(Some(split)) => split,
        Ok(None) => return Masses::Absent("no selection split published".into()),
        Err(error) => return Masses::Absent(format!("selection split: {error}")),
    };
    let change =
        split.smooth_mass + split.change_point_mass.iter().map(|(_, mass)| mass).sum::<f64>();
    assert!(
        (change + split.no_change_mass - 1.0).abs() < 1e-9,
        "the specification masses must sum to one"
    );
    let all_draws = match &result.fit {
        JointPanelFit::Varying(fit) => fit.draws.iter().all(Option::is_some),
        JointPanelFit::Invariant(_) => false,
    };
    Masses::Published {
        change,
        point_change: split.point_choice != TemporalModelChoice::NoChange,
        all_draws,
    }
}

/// Draws of the screening stage of the false-change measurement: the first draws of the
/// published whole-procedure run (the draws are indexed by their slot, so these are its first
/// ten).
const SCREEN_DRAWS: usize = 10;
/// A replicate is screened out when at most this share of its screening draws selects a change.
const SCREEN_MAX_CHANGE: f64 = 0.55;

/// The masses of one replicate, running the published `FREQUENTIST_DRAWS` draws only when the
/// first `SCREEN_DRAWS` of them leave a false confident change possible.
///
/// What the statistic is. A replicate errs when the published mass on specifications with a
/// change is at least the level: at least 95 of the 100 bootstrap selections choose a change
/// (every draw slot completed). Unlike a coverage indicator this is a function of the bootstrap
/// draws themselves, and the law of those draws given the data varies from order one across
/// replicates (a point fit that selected no change simulates from a no-change model and
/// selects a change rarely; one that selected a large change selects it almost always), so the
/// pooled-draw construction of the contrast tests does not apply: the error rate would be the
/// binomial tail at the pooled selection frequency.
///
/// What is done instead is sequential: the first `SCREEN_DRAWS` draws are the first slots of the
/// published run, and a replicate whose screen completed every slot, published its masses and
/// selected a change in at most `SCREEN_MAX_CHANGE` of them is scored as not erring without the
/// remaining draws. That conclusion is wrong only when the selection frequency `p` of the
/// replicate is high enough for 95 changes in 100 draws (probability `P(Bin(100, p) >= 95)`)
/// while at most 5 of the first 10 draws selected none: over every `p` the product of the two
/// probabilities is below 2e-4, so the screened error count is unbiased to that order and its
/// variance is that of the nested measurement. Anything else (a screen that failed a slot,
/// published no masses, or leaned to a change) runs the published draws and is scored exactly
/// as the nested measurement scores it.
fn screened_masses_reading(cell: Cell, mode: Mode, groups: usize, seed: u64) -> Masses {
    match masses_reading(cell, mode, groups, seed, SCREEN_DRAWS) {
        Masses::Published { change, point_change, all_draws: true }
            if change <= SCREEN_MAX_CHANGE =>
        {
            Masses::Screened { point_change }
        }
        _ => masses_reading(cell, mode, groups, seed, FREQUENTIST_DRAWS),
    }
}

/// Whether a replicate counts as a false confident change: `None` for a refused fit (a skip).
fn false_confident_change(reading: &Masses) -> Option<bool> {
    match reading {
        Masses::Refused(_) => None,
        Masses::Absent(_) => Some(true),
        Masses::Published { change, .. } => Some(*change >= LEVEL),
        Masses::Screened { .. } => Some(false),
    }
}

/// One-sided false-change measurement on panels whose mechanism never changes.
///
/// The published masses are selection frequencies of the bootstrap. Their calibration as
/// probabilities holds only for truths drawn from a prior (the Bayesian counterpart is scored
/// under prior-drawn truths in the posterior calibration tests), so a fixed no-change law cannot test that the mass
/// on "no change" equals a frequency. What a fixed no-change law can test, and what a reader
/// does with these masses, is the confident claim: a replicate scores a miss when the law
/// puts a mass of at least the level on specifications with a change, a false confident
/// change. The record asserts that false confident changes are at most `1 - level` of the
/// replicates: a floor on the complement, never an upper band. The point fit's own false
/// selection rate and the mean published change mass are printed beside it.
///
/// `screened` runs the published draws only for replicates whose first draws leave a false
/// confident change possible (`screened_masses_reading`); the mean mass is then printed over
/// the replicates that ran them.
fn measure_false_change(
    test: &'static str,
    cell: Cell,
    mode: Mode,
    seed_base: u64,
    screened: bool,
) {
    let groups = Path::Unchanged.groups();
    let readings = map_replicates(n_sim(), |rep| {
        if screened {
            screened_masses_reading(cell, mode, groups, seed_base + rep)
        } else {
            masses_reading(cell, mode, groups, seed_base + rep, FREQUENTIST_DRAWS)
        }
    });
    let mut tally = CoverageTally::new(test, LEVEL);
    let (mut published, mut point_changes, mut mass_sum) = (0_usize, 0_usize, 0.0);
    let mut screened_out = 0_usize;
    let mut refusals = BTreeMap::<String, usize>::new();
    for reading in &readings {
        match reading {
            Masses::Refused(reason) => {
                tally.skip();
                *refusals.entry(reason.clone()).or_default() += 1;
            }
            Masses::Absent(reason) => {
                // An executed fit that published no masses has not avoided a false claim
                // with a published statement: scored as a miss.
                tally.record(None, 0.5);
                *refusals.entry(reason.clone()).or_default() += 1;
            }
            Masses::Screened { point_change } => {
                screened_out += 1;
                point_changes += usize::from(*point_change);
                tally.record(Some((0.0, 1.0)), 0.5);
            }
            Masses::Published { change, point_change, .. } => {
                published += 1;
                point_changes += usize::from(*point_change);
                mass_sum += change;
                // The unit interval holds 0.5 and not 2.
                tally.record(Some((0.0, 1.0)), if *change < LEVEL { 0.5 } else { 2.0 });
            }
        }
    }
    eprintln!(
        "{test}: {published} replicate(s) published masses from the full draws and {screened_out} \
         were screened out by their first {SCREEN_DRAWS} draws; the point fit selected a change \
         in {point_changes}; mean published change mass over the full runs {:.4}",
        mass_sum / published.max(1) as f64
    );
    for (reason, count) in &refusals {
        eprintln!("{test}: {count} replicate(s) without masses: {reason}");
    }
    tally.assert_at_least();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn joint_panel_selection_false_change_frequentist() {
    measure_false_change(
        "joint_panel_selection_false_change_frequentist",
        LAGGED,
        Mode::Frequentist,
        33_000_000,
        !nested_requested(),
    );
}

/// The screened false-change measurement against the nested one on the same replicates: the
/// screen may differ from the published run only with probability below 2e-4 per replicate,
/// so the two error counts must agree replicate by replicate.
#[test]
#[ignore = "warp-speed validation: runs the nested measurement on 100 replicates; run by hand"]
fn joint_panel_warp_speed_validation_false_change() {
    let (cell, mode, seed_base) = (LAGGED, Mode::Frequentist, 33_000_000);
    let groups = Path::Unchanged.groups();
    let nested = map_replicates_from(0, VALIDATION_REPLICATES, |rep| {
        masses_reading(cell, mode, groups, seed_base + rep, FREQUENTIST_DRAWS)
    });
    let screened = map_replicates_from(0, VALIDATION_REPLICATES, |rep| {
        screened_masses_reading(cell, mode, groups, seed_base + rep)
    });
    let (mut errors_nested, mut errors_screened, mut disagree, mut skipped_out) =
        (0_u32, 0_u32, 0_u32, 0_u32);
    for (nested, screened) in nested.iter().zip(&screened) {
        skipped_out += u32::from(matches!(screened, Masses::Screened { .. }));
        let (a, b) = (false_confident_change(nested), false_confident_change(screened));
        errors_nested += u32::from(a == Some(true));
        errors_screened += u32::from(b == Some(true));
        disagree += u32::from(a != b);
    }
    eprintln!(
        "false-change screen validation: {errors_nested} false confident change(s) nested, \
         {errors_screened} screened, {disagree} replicate(s) differ, {skipped_out} of \
         {VALIDATION_REPLICATES} screened out"
    );
    assert_eq!(disagree, 0, "the screen must reproduce the nested error indicator");
}

// ---------------------------------------------------------------------------------------
// The experiment update and the value of a future sample.
// ---------------------------------------------------------------------------------------

/// A fitted Bayesian panel posterior, the experiment declaration over it and the retained
/// draws' value of the contrast.
struct Source {
    prepared: PreparedStudy<JointPanelState>,
    ledger: EvidenceLedger,
    /// Update with every field declared; only the observed mean changes between replicates.
    update: PanelExperimentUpdate,
    /// Contrast of every retained draw (the mapping is the identity) and its standard
    /// deviation.
    states: Vec<f64>,
    sd: f64,
    coefficients: Vec<f64>,
    /// Seed of the context the posterior was fitted under. The fitted source's dependency
    /// identity covers the execution seed, so the update must re-run the shared mechanism under
    /// this same context: a different seed is a different fitted source.
    fit_seed: u64,
}

impl Source {
    fn fit_context(&self) -> ExecutionContext {
        ExecutionContext::for_tests(grid_seed(self.fit_seed))
    }
}

/// One experimental observation has this multiple of the posterior standard deviation of the
/// contrast as its standard error: informative enough to move the weights, loose enough to
/// keep many draws at non-negligible weight.
const OBSERVATION_SD_RATIO: f64 = 0.8;
const EXPERIMENT_SAMPLE: u64 = 100;

fn fixture_ledger(original: &JointPanelResult) -> (EvidenceLedger, SourceId, SourceId, SourceId) {
    let mut ledger = EvidenceLedger::new();
    let model = ledger
        .add_source(
            SourceDeclaration::new(
                original.dependency_identity().to_hex(),
                SourceKind::FittedModel,
            )
            .with_observations([ObservationId::from_key("panel", b"source")]),
        )
        .unwrap();
    let mapping = ledger
        .add_source(
            SourceDeclaration::new("mapping", SourceKind::FittedModel)
                .with_observations([ObservationId::from_key("mapping", b"source")]),
        )
        .unwrap();
    let experiment = ledger
        .add_source(
            SourceDeclaration::new("geo_experiment", SourceKind::Experiment)
                .with_observations([ObservationId::from_key("geo", b"observed")]),
        )
        .unwrap();
    (ledger, model, mapping, experiment)
}

/// Identity affine bridge: the experimental mean is the contrast itself.
fn identity_bridge(target: &antecedent_contract::ScientificQuantity) -> BridgeContract {
    let mut experiment_quantity = target.clone();
    experiment_quantity.functional_id = "experimental_sample_mean".into();
    let setting = ExposureSetting {
        population_id: target.population_id.clone(),
        time_start: format!("program_period:{}", HORIZON - 1),
        time_end: format!("program_period:{}", HORIZON - 1),
        dose: 1.0,
        dose_units: "declared_joint_program".into(),
        duration: 1.0,
        duration_units: "panel_period".into(),
        exposure_regime: target.regime_id.clone(),
    };
    let bridge = BridgeContract {
        kind: BridgeKind::ParameterInterventionResponse,
        source: BridgeEndpoint {
            quantity: experiment_quantity,
            class: QuantityClass::ModelParameter,
        },
        target: BridgeEndpoint {
            quantity: target.clone(),
            class: QuantityClass::InterventionResponse,
        },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope {
            population_id: target.population_id.clone(),
            regime_id: target.regime_id.clone(),
        },
        mapping: BridgeMapping::Affine { intercept: 0.0, slope: 1.0 },
        assumptions: vec![BridgeAssumption {
            id: "declared_experiment_likelihood".into(),
            kind: AssumptionKind::ParameterInterpretation,
            statement: "The experimental Gaussian mean is this exact future joint-response contrast under frozen modular conditioning".into(),
            evidence: Some("geo_experiment".into()),
        }],
        transport: None,
        mapping_uncertainty: Some(MappingUncertainty {
            id: "paired_mapping".into(),
            intercept_variance: 0.0,
            slope_variance: 0.0,
            intercept_slope_covariance: 0.0,
        }),
        source_dependence: Some(SourceMappingDependence::SlopeCovariance {
            covariance: 0.0,
            evidence: "retained_joint_slots".into(),
        }),
        estimand_bridge: Some(EstimandBridge {
            id: "exact_program_translation".into(),
            experiment_id: "geo_experiment".into(),
            experiment: setting.clone(),
            target: setting,
            recency_delta: 0.0,
            recency_units: "panel_period".into(),
            spend_delta: 0.0,
            spend_units: "declared_joint_program".into(),
        }),
    };
    bridge.check_support().unwrap();
    bridge
}

/// Fit one Bayesian panel posterior and declare the experiment over its terminal contrast.
fn experiment_source(cell: Cell, path: Path, seed: u64) -> Source {
    const SOURCE_ATTEMPTS: u64 = 6;
    let groups = path.groups();
    let functional = JointPanelResponseFunctional::TerminalMean;
    // The source posterior must pass its declared diagnostics before an experiment can update
    // it. One fit serves every replicate, so it gets generous effort, and a posterior that
    // misses the thresholds is refitted on a fresh panel a bounded number of times, with the
    // diagnostics printed, before the test fails.
    let mut source = None;
    for attempt in 0..SOURCE_ATTEMPTS {
        let attempt_seed = seed + 7_919 * attempt;
        let generated = generate_panel(cell, path, groups, attempt_seed);
        let sampling = PanelSamplingOptions {
            chains: 4,
            warmup: 5_000,
            draws: 5_000,
            ..PanelSamplingOptions::default()
        };
        let fitted =
            fit_generated(cell, path, Mode::Bayesian, sampling, groups, generated, attempt_seed);
        let (prepared, result, _, _, _) = match fitted {
            Ok(fitted) => fitted,
            Err(reason) => {
                eprintln!("source attempt {attempt}: the fit was refused: {reason}");
                continue;
            }
        };
        match result.numerical_project_law_source(&functional) {
            Ok(law) => {
                source = Some((prepared, result, law, attempt_seed));
                break;
            }
            Err(error) => {
                let diagnostics = match &result.fit {
                    JointPanelFit::Invariant(fit) => format!("{:?}", fit.diagnostics),
                    JointPanelFit::Varying(fit) => format!(
                        "posterior diagnostics [rhat, bulk ess, tail ess] {:?}, converged {:?}",
                        fit.posterior_diagnostics, fit.posterior_converged
                    ),
                };
                eprintln!("source attempt {attempt}: no complete law ({error}); {diagnostics}");
            }
        }
    }
    let (prepared, result, prior, fit_seed) = source.unwrap_or_else(|| {
        panic!("no source posterior passed its diagnostics in {SOURCE_ATTEMPTS} attempts")
    });
    let width = prior.shape()[1];
    let TargetForm::Contrast(contrast) = result.request().target.form() else {
        panic!("assembled target")
    };
    let mut coefficients = vec![0.0; width];
    for term in contrast.terms() {
        let index =
            result.request().worlds.iter().position(|world| world.world == term.world).unwrap();
        coefficients[index] = term.coefficient;
    }
    let states: Vec<f64> = prior
        .draws()
        .chunks_exact(width)
        .map(|row| row.iter().zip(&coefficients).map(|(value, weight)| value * weight).sum())
        .collect();
    let n = states.len();
    let mean = states.iter().sum::<f64>() / n as f64;
    let sd = (states.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
    assert!(sd > 0.0, "the retained posterior must carry uncertainty in its contrast");
    let target = result.experiment_quantity(&functional).unwrap();
    let (ledger, model, mapping, experiment) = fixture_ledger(&result);
    let update = PanelExperimentUpdate {
        bridge: identity_bridge(&target),
        functional,
        observed_mean: mean,
        observation_variance: EXPERIMENT_SAMPLE as f64 * (OBSERVATION_SD_RATIO * sd).powi(2),
        sample_size: EXPERIMENT_SAMPLE,
        modular_policy: PanelExperimentModularPolicy::FrozenJointImportance,
        mapping: PanelExperimentMapping {
            dependency: result.dependency_identity(),
            intercepts: vec![0.0; n],
            slopes: vec![1.0; n],
        },
        evidence: PanelExperimentEvidence { model, prior_origins: vec![], experiment, mapping },
    };
    Source { prepared, ledger, update, states, sd, coefficients, fit_seed }
}

/// Equal-tailed interval of an atomic law: the first atoms whose cumulative weight reaches
/// each tail.
fn weighted_interval(states: &[f64], order: &[usize], weights: &[f64]) -> Option<(f64, f64)> {
    let total: f64 = weights.iter().sum();
    let tail = (1.0 - LEVEL) / 2.0;
    let (mut cumulative, mut low, mut high) = (0.0, None, None);
    for &index in order {
        cumulative += weights[index] / total;
        if low.is_none() && cumulative >= tail {
            low = Some(states[index]);
        }
        if cumulative >= 1.0 - tail {
            high = Some(states[index]);
            break;
        }
    }
    low.zip(high)
}

fn experiment_construction(cell: Cell, path: Path, interval: &str) -> Construction {
    Construction {
        query: "JointPanelExperimentUpdate".into(),
        graph_class: "NotApplicable".into(),
        structure: "panel".into(),
        modality: "experiment_update".into(),
        inference: "Bayesian".into(),
        estimator: format!(
            "joint_panel.frozen_joint_importance.{}.{}",
            if cell.lagged { "finite_lag" } else { "static" },
            if path.varying() { "time_varying_selection" } else { "invariant" },
        ),
        interval_method: interval.into(),
        se_kind: String::new(),
        dependence: "finite_retained_posterior".into(),
        posterior: "importance_weighted_retained_draws".into(),
        functional: "terminal_interaction_contrast".into(),
        identification: "model_conditional".into(),
        reported_level: LEVEL,
    }
}

/// Calibration of the update's weights: the truth is one retained draw, drawn uniformly (the
/// law the update treats as its prior), the experiment is simulated from it under the
/// declared Gaussian likelihood, and the equal-tailed interval of the updated weights is
/// scored. Under exact conditioning this is exactly calibrated whatever the retained law is,
/// so a miss is an error of the likelihood, the mapping or the weight arithmetic, and the
/// published posterior moments of any quantity are moments of these same weights. It does not
/// measure whether the retained draws represent the panel posterior; the fitted-panel
/// contrast intervals measure that.
fn measure_update_weights(test: &'static str, cell: Cell, path: Path, seed: u64) {
    let source = experiment_source(cell, path, seed);
    let n = source.states.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| source.states[a].total_cmp(&source.states[b]));
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp: "experiment_source", interval: "weighted_posterior_quantile" },
        LEVEL,
    );
    // The two generators are consumed in replicate order, so every replicate's draw is taken
    // up front and the replicates themselves are independent of execution order.
    let mut draw = uniform(grid_seed(seed), 41);
    let mut noise = gaussian(stream_seed(grid_seed(seed), 42));
    let inputs: Vec<(usize, f64)> =
        (0..n_sim()).map(|_| (((draw() * n as f64) as usize).min(n - 1), noise())).collect();
    let mut refused = BTreeMap::<String, usize>::new();
    let readings = map_replicates(n_sim(), |rep| {
        let (slot, shock) = inputs[usize::try_from(rep).expect("replicate index")];
        let truth = source.states[slot];
        let mut update = source.update.clone();
        update.observed_mean = truth + OBSERVATION_SD_RATIO * source.sd * shock;
        let ctx = source.fit_context();
        let attempt = source
            .prepared
            .with_experiment(update, &source.ledger)
            .and_then(|program| program.estimate(&source.ledger, &ctx));
        let updated = match attempt {
            Ok(updated) => updated,
            Err(error) => return Err(error.to_string()),
        };
        let weights: Option<Vec<f64>> = updated
            .numerical_law_source(&source.ledger)
            .ok()
            .and_then(|law| law.metadata().weights.clone());
        Ok((weights.and_then(|weights| weighted_interval(&source.states, &order, &weights)), truth))
    });
    for reading in readings {
        let (interval, truth) = match reading {
            Ok(reading) => reading,
            Err(error) => {
                tally.skip();
                *refused.entry(error).or_default() += 1;
                continue;
            }
        };
        tally.bind_validation(
            &experiment_construction(cell, path, "weighted_posterior_quantile"),
            ScopeFacts {
                row_count: path.groups() as u64,
                replicates_ok: None,
                posterior_draws: Some(u32::try_from(n).unwrap()),
                unidentified_mass: 0.0,
            },
            "not_applicable",
        );
        tally.record(interval, truth);
    }
    for (reason, count) in &refused {
        eprintln!("{test}: {count} replicate(s) refused: {reason}");
    }
    tally.assert();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn joint_panel_experiment_update_weights_finite_lag_bayesian_l95() {
    measure_update_weights(
        "joint_panel_experiment_update_weights_finite_lag_bayesian_l95",
        LAGGED,
        Path::Invariant,
        34_000_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn joint_panel_experiment_update_weights_time_varying_bayesian_l95() {
    measure_update_weights(
        "joint_panel_experiment_update_weights_time_varying_bayesian_l95",
        LAGGED,
        Path::Changed,
        34_100_000,
    );
}

/// Terminal utility of the two actions: hold (zero) and change (the contrast above a
/// threshold).
struct ContrastUtility {
    coefficients: Vec<f64>,
    threshold: f64,
}
impl Utility<usize, PanelExperimentDecisionState> for ContrastUtility {
    fn evaluate_batch(
        &self,
        actions: &[usize],
        states: &[PanelExperimentDecisionState],
        out: &mut [f64],
    ) -> Result<(), DesignError> {
        for (action_index, action) in actions.iter().enumerate() {
            for (slot, state) in states.iter().enumerate() {
                out[action_index * states.len() + slot] = if *action == 0 {
                    0.0
                } else {
                    state
                        .responses
                        .iter()
                        .zip(&self.coefficients)
                        .map(|(value, weight)| value * weight)
                        .sum::<f64>()
                        - self.threshold
                };
            }
        }
        Ok(())
    }
}

/// Exact value of a next study of `sigma` standard error over an atomic belief, by
/// one-dimensional quadrature of the predictive mixture, and its quadrature error bound. The
/// integrand is `max(0, sum_i w_i N(y; m_i, sigma^2) (q_i - threshold))`; no library signal,
/// update, utility or value helper is called.
fn exact_study_value(states: &[f64], weights: &[f64], threshold: f64, sigma: f64) -> (f64, f64) {
    const CELLS: u32 = 4_096;
    let total: f64 = weights.iter().sum();
    let atoms: Vec<(f64, f64)> = states
        .iter()
        .zip(weights)
        .map(|(state, weight)| (*state, weight / total * (state - threshold)))
        .collect();
    let lower = atoms.iter().map(|(m, _)| *m).fold(f64::INFINITY, f64::min) - 8.0 * sigma;
    let upper = atoms.iter().map(|(m, _)| *m).fold(f64::NEG_INFINITY, f64::max) + 8.0 * sigma;
    let step = (upper - lower) / f64::from(CELLS);
    let density_scale = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
    let mut future_value = 0.0;
    for cell in 0..CELLS {
        let y = lower + (f64::from(cell) + 0.5) * step;
        let advantage = atoms
            .iter()
            .map(|(m, gain)| gain * density_scale * (-0.5 * ((y - m) / sigma).powi(2)).exp())
            .sum::<f64>();
        future_value += step * advantage.max(0.0);
    }
    let current_value = atoms.iter().map(|(_, gain)| gain).sum::<f64>().max(0.0);
    let absolute_gain = atoms.iter().map(|(_, gain)| gain.abs()).sum::<f64>();
    // Midpoint error is at most h/2 times the total variation; the positive part is
    // Lipschitz and the L1 norm of a Normal density's derivative is sqrt(2/pi)/sigma. Mills'
    // inequality bounds both omitted tails.
    let variation = step / 2.0 * absolute_gain * (2.0 / std::f64::consts::PI).sqrt() / sigma;
    let tails =
        absolute_gain * 2.0 * (-32.0_f64).exp() / (8.0 * (2.0 * std::f64::consts::PI).sqrt());
    (future_value - current_value, variation + tails + 1e-10)
}

/// Monte Carlo standard error of the value of a future sample.
///
/// The update is fixed (one observed mean); the value of a further sample of the same mapped
/// estimand has an exact value from quadrature over the updated weights. Each replicate
/// reruns the library's ranking at an independent seed and scores
/// `gross_evsi +- 1.96 standard_error` against that exact value, so the published standard
/// error is measured as the precision of the published estimate.
fn measure_future_sample_standard_error(test: &'static str, cell: Cell, path: Path, seed: u64) {
    const REPLICATES: usize = 200;
    const NEXT_SAMPLE: u64 = 4;
    let source = experiment_source(cell, path, seed);
    let n = source.states.len();
    let mean = source.states.iter().sum::<f64>() / n as f64;
    let mut update = source.update.clone();
    update.observed_mean = mean + OBSERVATION_SD_RATIO * source.sd;
    let ctx = source.fit_context();
    let updated = source
        .prepared
        .with_experiment(update, &source.ledger)
        .and_then(|program| program.estimate(&source.ledger, &ctx))
        .unwrap_or_else(|error| panic!("the fixed update was refused: {error}"));
    let weights: Vec<f64> = updated
        .numerical_law_source(&source.ledger)
        .expect("the updated law")
        .metadata()
        .weights
        .as_ref()
        .expect("posterior weights")
        .clone();
    let total: f64 = weights.iter().sum();
    let threshold = source.states.iter().zip(&weights).map(|(s, w)| s * w / total).sum::<f64>();
    let observation_variance = NEXT_SAMPLE as f64 * (OBSERVATION_SD_RATIO * source.sd).powi(2);
    let sigma = (observation_variance / NEXT_SAMPLE as f64).sqrt();
    let (exact, quadrature_error) = exact_study_value(&source.states, &weights, threshold, sigma);
    eprintln!(
        "{test}: exact value of the next sample {exact:.6e} (quadrature error {quadrature_error:.2e})"
    );
    let problem = DecisionProblem::new(
        vec![0, 1],
        Arc::new(ContrastUtility { coefficients: source.coefficients.clone(), threshold }),
        vec![],
    );
    let candidate = PanelFutureSample {
        id: "next".into(),
        sample_size: NEXT_SAMPLE,
        observation_variance,
        cost_utility: Some(0.0),
    };
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp: "experiment_source", interval: "monte_carlo_standard_error" },
        LEVEL,
    );
    let readings = map_replicates(n_sim(), |rep| {
        let ctx = ExecutionContext::for_tests(grid_seed(seed + 1 + rep));
        updated
            .rank_next_samples(
                &problem,
                std::slice::from_ref(&candidate),
                REPLICATES,
                &source.ledger,
                &ctx,
            )
            .ok()
            .map(|ranked| (ranked[0].gross_evsi, ranked[0].standard_error))
    });
    for reading in readings {
        let Some((gross_evsi, standard_error)) = reading else {
            tally.skip();
            continue;
        };
        tally.bind_validation(
            &experiment_construction(cell, path, "monte_carlo_standard_error"),
            ScopeFacts {
                row_count: path.groups() as u64,
                replicates_ok: Some(u32::try_from(REPLICATES).unwrap()),
                posterior_draws: Some(u32::try_from(n).unwrap()),
                unidentified_mass: 0.0,
            },
            "not_applicable",
        );
        tally.record(normal_interval(gross_evsi, Some(standard_error), Z95), exact);
    }
    tally.assert();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn joint_panel_future_sample_value_standard_error_finite_lag_bayesian_l95() {
    measure_future_sample_standard_error(
        "joint_panel_future_sample_value_standard_error_finite_lag_bayesian_l95",
        LAGGED,
        Path::Invariant,
        35_000_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn joint_panel_future_sample_value_standard_error_time_varying_bayesian_l95() {
    measure_future_sample_standard_error(
        "joint_panel_future_sample_value_standard_error_time_varying_bayesian_l95",
        LAGGED,
        Path::Changed,
        35_100_000,
    );
}

// ---------------------------------------------------------------------------------------
// The prepared Gaussian joint response.
// ---------------------------------------------------------------------------------------

/// Gaussian-linear response, `y = 3 + 2 a - 1.5 b + 0.5 z + e` with `a`, `b` independent on
/// `{-1, 0, 1}`, `z` and `e` standard normal. The contrast of the worlds `(a, b) = (1, 1)`
/// and `(0, 0)` is `2 - 1.5 = 0.5` exactly: the fitted mean is additive, so it does not
/// depend on the empirical covariates it conditions on.
const RESPONSE_TRUTH: f64 = 0.5;

fn response_data(n: usize, seed: u64) -> TabularData {
    let mut normal = gaussian(stream_seed(seed, 1));
    let mut unit = uniform(stream_seed(seed, 2), 5);
    let mut columns = [vec![], vec![], vec![], vec![]];
    for _ in 0..n {
        let a = f64::from((unit() * 3.0) as u8) - 1.0;
        let b = f64::from((unit() * 3.0) as u8) - 1.0;
        let z = normal();
        let y = 3.0 + 2.0 * a - 1.5 * b + 0.5 * z + normal();
        for (column, value) in columns.iter_mut().zip([a, b, y, z]) {
            column.push(value);
        }
    }
    TabularData::from_f64_columns(
        ["a", "b", "y", "z"].into_iter().zip(columns.iter().map(Vec::as_slice)),
    )
    .unwrap()
}

fn response_target() -> ScientificTarget {
    let binding = WorldBinding::new(
        TargetPopulation::AllObserved,
        ObservationUnit::new("row").unwrap(),
        OutcomeQuantity::mean(VariableId::from_raw(2), QuantityUnits::new("unspecified").unwrap())
            .unwrap(),
        Conditioning::unconditional(),
    )
    .unwrap();
    let worlds = WorldCollection::new([(0.0, 0.0), (1.0, 1.0)].into_iter().enumerate().map(
        |(index, (a, b))| {
            InterventionWorld::new(
                ScientificWorldId::from_raw(u32::try_from(index).unwrap()),
                WorldAction::at_decision([
                    Intervention::set(VariableId::from_raw(0), Value::f64(a)),
                    Intervention::set(VariableId::from_raw(1), Value::f64(b)),
                ])
                .unwrap(),
                index == 0,
                binding.clone(),
                TimeFrame::static_decision(),
            )
            .unwrap()
        },
    ))
    .unwrap();
    let id = ScientificWorldId::from_raw;
    ScientificTarget::new(
        TargetForm::Contrast(
            InterventionContrast::difference(worlds, id(1), id(0), id(0)).unwrap(),
        ),
        ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
        InferenceSelection::Unspecified,
    )
}

/// Scored reading of the prepared response contrast in one inference mode.
#[allow(
    clippy::option_option,
    reason = "a deliberate tri-state: no reading when the study refuses, then an interval that may be absent"
)]
fn response_reading(mode: Mode, n: usize, seed: u64) -> Option<Option<(f64, f64)>> {
    let data = response_data(n, seed);
    let ctx = ExecutionContext::for_tests(grid_seed(seed));
    let graph =
        Dag::from_named_edges(data.schema(), &[("a", "y"), ("b", "y"), ("z", "y")]).unwrap();
    let query = antecedent_core::ResponseQuery::new(
        antecedent_core::ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(2),
            interventions: Arc::from([
                Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
                Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
            ]),
        },
    );
    let inference = match mode {
        Mode::Frequentist => InferenceMode::Frequentist,
        Mode::Bayesian => InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(2_000)),
    };
    let prepared = Study::tabular(data)
        .graph(graph)
        .query(query)
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .ok()?
        .prepare(&ctx)
        .ok()?;
    let program = prepared.compile_joint_response(&response_target(), &ctx).ok()?;
    let result = program.execute(&ctx).ok()?;
    let evaluation = result.evaluation();
    Some(match mode {
        Mode::Frequentist => evaluation
            .contrast
            .and_then(|contrast| normal_interval(contrast.value, Some(contrast.se), Z95)),
        Mode::Bayesian => evaluation.contrast_draws.as_ref().and_then(|draws| {
            let finite: Vec<f64> = draws.iter().copied().filter(|v| v.is_finite()).collect();
            quantile_interval(&finite, LEVEL)
        }),
    })
}

fn measure_response(test: &'static str, mode: Mode, seed_base: u64) {
    let n = grid_n(300);
    let interval = match mode {
        Mode::Frequentist => "analytic_se",
        Mode::Bayesian => "posterior_quantile",
    };
    let readings = map_replicates(n_sim(), |rep| response_reading(mode, n, seed_base + rep));
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp: "response_data", interval }, LEVEL);
    for reading in readings {
        let Some(interval_read) = reading else {
            tally.skip();
            continue;
        };
        tally.bind_validation(
            &Construction {
                query: "JointResponse".into(),
                graph_class: "Dag".into(),
                structure: "fixed".into(),
                modality: "tabular".into(),
                inference: match mode {
                    Mode::Frequentist => "Frequentist",
                    Mode::Bayesian => "Bayesian",
                }
                .into(),
                estimator: "joint_response.gaussian_linear".into(),
                interval_method: interval.into(),
                se_kind: match mode {
                    Mode::Frequentist => "homoskedastic_ols",
                    Mode::Bayesian => "",
                }
                .into(),
                dependence: "iid".into(),
                posterior: match mode {
                    Mode::Frequentist => "",
                    Mode::Bayesian => "conjugate_gaussian",
                }
                .into(),
                functional: "contrast_of_world_means".into(),
                identification: "point".into(),
                reported_level: LEVEL,
            },
            ScopeFacts {
                row_count: n as u64,
                replicates_ok: None,
                posterior_draws: (mode == Mode::Bayesian).then_some(2_000),
                unidentified_mass: 0.0,
            },
            "not_applicable",
        );
        tally.record(interval_read, RESPONSE_TRUTH);
    }
    tally.assert();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn prepared_joint_response_contrast_frequentist_l95() {
    measure_response(
        "prepared_joint_response_contrast_frequentist_l95",
        Mode::Frequentist,
        36_000_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn prepared_joint_response_contrast_bayesian_l95() {
    measure_response("prepared_joint_response_contrast_bayesian_l95", Mode::Bayesian, 36_100_000);
}

// ---------------------------------------------------------------------------------------
// Whole-search reliability of the selected program.
// ---------------------------------------------------------------------------------------
//
// The continuous program search polls programs on the one fitted response law (the frozen fit's
// aligned draws) and returns the best it found. Its published reliability is the simultaneous
// interval for the selected program's value over every retained proposal
// (`ContinuousSearchReliability::Interval`). Every test repeats the whole procedure on
// independent panels: data generation, the fit, the search and the interval, and scores the
// interval against the value of the program the search returned under the generating law.
//
// Truth. The search maximizes the terminal mean response `TerminalMean` of the reference world
// with the last periods' doses of both channels as controls (the last period for the static
// cell, the last three for the carryover cell, where the exposure window of the terminal period
// lies wholly inside the program) under a total-dose budget. For a program with exposures
// `e_d` (the lag-weighted doses; the dose itself without carryover) the value over the panel's
// groups is closed form,
//
//   value(x) = mean_g [ c + b0 s0(e0) + b1 s1(e1) + b01 s0(e0) s1(e1) + gamma z_gT + season_gT + u_g ],
//
// with `u_g` the group's realized intercept and `s_d` the saturation. The best program is the
// maximizer of this function over the feasible doses; it needs no fit. Because every unit of
// dose costs the same and no lag weight exceeds the current period's, the best program puts all
// of each channel's dose at the terminal period, and the maximizer is found on a dense grid of
// the two terminal doses. The value of the *selected* program, not the best, is what the
// interval claims; the regret of the selection is reported next to it.
//
// The interval is a whole-procedure statement, so it must hold at least at its nominal level
// (it is simultaneous over the proposals, conservative by construction): the frequentist panels
// are fixed-truth, the Bayesian panels are simulation-based calibration (truths drawn from the
// fit's own proper prior), as for the contrast intervals above.

/// Total-dose budget of the searched programs.
const SEARCH_BUDGET: f64 = 8.0;
/// Largest dose of any scheduled control.
const SEARCH_DOSE_CAP: f64 = 8.0;
/// Proposals the search may evaluate. The step tolerance is far below any reachable step, so a
/// search always stops at this count: every replicate has the same candidate-set size.
const SEARCH_EVALUATIONS: usize = 40;

/// The generating law's value of a program over the realized groups of one panel.
struct ProgramTruth {
    law: Law,
    lagged: bool,
    /// Confounder, season and realized intercept of each group at the terminal period.
    groups: Vec<(f64, usize, f64)>,
}

impl ProgramTruth {
    fn of(law: Law, cell: Cell, generated: &Generated) -> Self {
        let groups = generated
            .scenarios
            .iter()
            .zip(&generated.intercepts)
            .map(|(scenario, &intercept)| {
                (scenario.confounder[HORIZON - 1], scenario.season[HORIZON - 1], intercept)
            })
            .collect();
        Self { law, lagged: cell.lagged, groups }
    }

    /// Lag-weighted doses of both channels at the terminal period.
    fn exposures(&self, controls: &[f64]) -> [f64; 2] {
        if self.lagged {
            // The doses of a channel are the periods HORIZON-3, HORIZON-2, HORIZON-1.
            let weighted = |channel: usize, doses: &[f64]| {
                self.law.lag[channel][0] * doses[2]
                    + self.law.lag[channel][1] * doses[1]
                    + self.law.lag[channel][2] * doses[0]
            };
            [weighted(0, &controls[0..3]), weighted(1, &controls[3..6])]
        } else {
            [controls[0], controls[1]]
        }
    }

    /// Value of the program with these scheduled doses.
    fn value(&self, controls: &[f64]) -> f64 {
        let law = &self.law;
        let [e0, e1] = self.exposures(controls);
        let (s0, s1) = (law.saturate(0, 0, e0), law.saturate(0, 1, e1));
        let terminal = law.intercept
            + law.channel[0] * s0
            + law.channel[1] * s1
            + law.interaction_at(HORIZON - 1) * s0 * s1;
        self.groups
            .iter()
            .map(|&(z, season, intercept)| {
                terminal + law.confounder * z + law.season[season] + intercept
            })
            .sum::<f64>()
            / self.groups.len() as f64
    }

    /// The best feasible program's value, by a dense grid of the two terminal doses.
    fn best_value(&self) -> f64 {
        let steps = 400_usize;
        let width = SEARCH_DOSE_CAP.min(SEARCH_BUDGET) / steps as f64;
        let mut best = f64::NEG_INFINITY;
        for i in 0..=steps {
            for j in 0..=steps {
                let (a, b) = (width * i as f64, width * j as f64);
                if a + b > SEARCH_BUDGET + 1e-12 {
                    continue;
                }
                let controls =
                    if self.lagged { vec![0.0, 0.0, a, 0.0, 0.0, b] } else { vec![a, b] };
                best = best.max(self.value(&controls));
            }
        }
        best
    }
}

/// One replicate: the interval the search published for its selected program, the truth value of
/// that program and the construction the runtime keyed the interval under.
enum SearchReading {
    Scored {
        interval: Option<(f64, f64)>,
        truth: f64,
        /// `true best value - true value of the selected program`, at least zero.
        regret: f64,
        bound: Option<(Construction, ScopeFacts, String)>,
    },
    Refused(String),
}

fn search_domain(lagged: bool) -> (FiniteProgramDomain, usize) {
    let offsets: Vec<i32> = if lagged {
        (HORIZON - 3..HORIZON).map(|offset| i32::try_from(offset).unwrap()).collect()
    } else {
        vec![i32::try_from(HORIZON - 1).unwrap()]
    };
    let controls: Vec<ProgramControl> = (0..2_u32)
        .flat_map(|channel| {
            offsets.iter().map(move |&offset| ProgramControl {
                variable: VariableId::from_raw(channel),
                offset,
                lower: 0.0,
                upper: SEARCH_DOSE_CAP,
                unit_cost: 1.0,
            })
        })
        .collect();
    let n = controls.len();
    (FiniteProgramDomain { controls, budget: Some(SEARCH_BUDGET), restrictions: vec![] }, n)
}

fn search_reading(
    fitted: Result<Fitted, String>,
    truth: &ProgramTruth,
    best_value: f64,
) -> SearchReading {
    let (_, result, _, _, ctx) = match fitted {
        Ok(fitted) => fitted,
        Err(reason) => return SearchReading::Refused(reason),
    };
    let (domain, n) = search_domain(truth.lagged);
    let search = match result.compile_continuous_search(
        JointPanelContinuousSearchRequest {
            base_world: ScientificWorldId::from_raw(0),
            scope: JointPanelScientificScope::ModelConditional,
            functional: JointPanelResponseFunctional::TerminalMean,
            uncertainty: JointPanelSearchUncertainty::AlignedLaw,
            contract: ProgramSearchContract {
                space: ProgramSearchSpace {
                    id: "selected_program_reliability".into(),
                    variables: vec!["response".into()],
                },
                objective: ProgramObjective::ExpectedUtility(UtilityExpr::Input(0)),
                constraints: vec![],
            },
            domain,
            utility_units: "outcome".into(),
            options: ContinuousSearchOptions {
                max_evaluations: SEARCH_EVALUATIONS,
                max_iterations: 10_000,
                initial_step: 0.25,
                step_tolerance: 1e-12,
                objective_tolerance: 1e-12,
            },
            initial: vec![0.0; n],
        },
        &ctx,
    ) {
        Ok(search) => search,
        Err(error) => return SearchReading::Refused(format!("compile search: {error}")),
    };
    let found = match search.execute(&ctx) {
        Ok(found) => found,
        Err(error) => return SearchReading::Refused(format!("execute search: {error}")),
    };
    let Some(best) = found.search.best.as_ref() else {
        return SearchReading::Refused("no feasible program".into());
    };
    let selected_value = truth.value(&best.controls);
    let interval = found.search.reliability.interval();
    let bound = interval
        .and_then(|interval| selected_value_basis(&interval.coordinate, LEVEL))
        .map(|basis| {
            let key = basis.key;
            (
                Construction {
                    query: key.query,
                    graph_class: key.graph_class,
                    structure: key.structure,
                    modality: key.modality,
                    inference: key.inference,
                    estimator: key.estimator,
                    interval_method: key.interval_method,
                    se_kind: key.se_kind,
                    dependence: key.dependence,
                    posterior: key.posterior,
                    functional: key.functional,
                    identification: key.identification,
                    reported_level: key.level,
                },
                ScopeFacts {
                    row_count: basis.scope.row_count,
                    replicates_ok: basis.scope.replicates_ok,
                    posterior_draws: basis.scope.posterior_draws,
                    unidentified_mass: basis.scope.unidentified_mass,
                },
                key.validation,
            )
        });
    SearchReading::Scored {
        interval: interval.map(|interval| (interval.lower, interval.upper)),
        truth: selected_value,
        regret: (best_value - selected_value).max(0.0),
        bound,
    }
}

fn frequentist_search_reading(cell: Cell, seed: u64) -> SearchReading {
    let groups = Path::Invariant.groups();
    let generated = generate_panel(cell, Path::Invariant, groups, seed);
    let truth = ProgramTruth::of(Law::of(cell), cell, &generated);
    let best = truth.best_value();
    let fitted = fit_generated(
        cell,
        Path::Invariant,
        Mode::Frequentist,
        default_sampling(),
        groups,
        generated,
        seed,
    );
    search_reading(fitted, &truth, best)
}

fn bayesian_search_reading(cell: Cell, seed: u64) -> SearchReading {
    let (law, _) = prior_law(cell, false, seed);
    let generated = generate_from(cell, &law, HORIZON, CALIBRATION_GROUPS, seed);
    let truth = ProgramTruth::of(law, cell, &generated);
    let best = truth.best_value();
    let fitted = fit_generated(
        cell,
        Path::Invariant,
        Mode::Bayesian,
        default_sampling(),
        CALIBRATION_GROUPS,
        generated,
        seed,
    );
    search_reading(fitted, &truth, best)
}

fn measure_selected_program(
    test: &'static str,
    dgp: &'static str,
    cell: Cell,
    mode: Mode,
    seed_base: u64,
) {
    let readings = map_replicates(n_sim(), |rep| match mode {
        Mode::Frequentist => frequentist_search_reading(cell, seed_base + rep),
        Mode::Bayesian => bayesian_search_reading(cell, seed_base + rep),
    });
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp, interval: SELECTED_VALUE_INTERVAL_METHOD },
        LEVEL,
    );
    let mut refusals = BTreeMap::<String, usize>::new();
    let (mut withheld, mut regret_sum, mut scored) = (0_usize, 0.0, 0_usize);
    for reading in &readings {
        match reading {
            SearchReading::Refused(reason) => {
                tally.skip();
                *refusals.entry(reason.clone()).or_default() += 1;
            }
            SearchReading::Scored { interval, truth, regret, bound } => {
                withheld += usize::from(interval.is_none());
                regret_sum += regret;
                scored += 1;
                match bound {
                    Some((construction, scope, validation)) => {
                        tally.bind_validation(construction, *scope, validation);
                        tally.record(*interval, *truth);
                    }
                    // No record can describe this coordinate: nothing to bind the score to.
                    None => tally.skip(),
                }
            }
        }
    }
    eprintln!(
        "{test}: {withheld} search(es) published no interval (scored as misses); mean regret of \
         the selected program against the true best {:.5} over {scored} replicates",
        regret_sum / scored.max(1) as f64
    );
    for (reason, count) in &refusals {
        eprintln!("{test}: {count} replicate(s) refused: {reason}");
    }
    tally.assert_one_sided(
        CoverageSide::Upper,
        "the simultaneous interval contains the value, under the generating law, of the program \
         the search selected, at least at the nominal level",
    );
}

macro_rules! selected_program_test {
    ($name:ident, $dgp:literal, $cell:expr, $mode:expr, $seed:expr) => {
        #[test]
        #[ignore = "coverage: measure with scripts/measure_calibration.sh"]
        fn $name() {
            measure_selected_program(stringify!($name), $dgp, $cell, $mode, $seed);
        }
    };
}

selected_program_test!(
    joint_panel_selected_program_value_static_frequentist_l95,
    "generate_panel",
    STATIC,
    Mode::Frequentist,
    33_000_000
);
selected_program_test!(
    joint_panel_selected_program_value_static_bayesian_l95,
    "generate_from",
    STATIC,
    Mode::Bayesian,
    33_200_000
);
selected_program_test!(
    joint_panel_selected_program_value_temporal_frequentist_l95,
    "generate_panel",
    LAGGED,
    Mode::Frequentist,
    33_400_000
);
selected_program_test!(
    joint_panel_selected_program_value_temporal_bayesian_l95,
    "generate_from",
    LAGGED,
    Mode::Bayesian,
    33_600_000
);

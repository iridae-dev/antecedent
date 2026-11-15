//! Coverage of the further statements a frequentist temporal design fit publishes: every
//! other coordinate's pointwise interval, the sup-t band (joint coverage of its coordinates),
//! the Holm-adjusted p-values (no rejection at 0.05 under the global null), quantile levels
//! beside the median, the local-quadratic derivative, the bias-aware interval under a
//! non-zero curvature bound, and regime survival at a longer horizon.
//!
//! Builders mirror the family's first harness file, parametrized where a statement needs it.
//! Records are keyed on the design axes (structure `design`, validation `not_applicable`). A
//! replicate that publishes nothing to score is a skip (a miss).
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent::{InferenceMode, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, TemporalAssignment, TemporalDesignFunctional,
    TemporalDesignQuery, TemporalDoseTarget, VariableId,
};
use antecedent_data::TabularData;
use common::calibration::{
    Construction, CoverageSide, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n,
    map_replicates, n_sim, uniform,
};
use common::calibration_bind::constructions;
use std::sync::{Arc, OnceLock};

/// Per-decision effect of the two-arm panel.
const TWO_ARM_EFFECT: f64 = 1.25;

/// Gaussian kernel bandwidth of every dose target.
const BANDWIDTH: f64 = 0.3;

/// Doses are uniform on `(0, DOSE_TOP)`.
const DOSE_TOP: f64 = 2.0;

/// Whole-procedure bootstrap replicates of the bootstrap routes: the route's replicate floor.
const BOOTSTRAP_REPLICATES: u32 = 99;

fn labels(values: &[String]) -> Arc<[Arc<str>]> {
    values.iter().map(|value| Arc::<str>::from(value.as_str())).collect::<Vec<_>>().into()
}

fn table(columns: &[(&str, &[f64])]) -> TabularData {
    TabularData::from_f64_columns(columns.iter().copied())
        .expect("the simulated columns are finite and aligned")
}

fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// One route of the family.
#[derive(Clone, Copy, Debug)]
enum Design {
    TwoArm,
    MultiArm,
    Categorical,
    Factorial,
    Interference,
    ContinuousDose,
    SmoothedDose,
    DoseGrid,
    Quantile,
}

impl Design {
    /// Units (or interference clusters, or subjects) at this run's sample-size grid point.
    fn units(self) -> usize {
        grid_n(match self {
            Self::MultiArm | Self::Categorical => 250,
            Self::Quantile => 200,
            _ => 300,
        })
    }

    /// Draw one panel and its query.
    fn draw(self, units: usize, seed: u64) -> (TabularData, TemporalDesignQuery) {
        match self {
            Self::TwoArm => two_arm(units, seed),
            Self::MultiArm => multi_arm(units, seed, false),
            Self::Categorical => multi_arm(units, seed, true),
            Self::Factorial => factorial(units, seed),
            Self::Interference => interference(units, seed),
            Self::ContinuousDose => dose(
                units,
                seed,
                TemporalDoseTarget::SmoothedContrast { active: 1.2, control: 0.8 },
            ),
            Self::SmoothedDose => {
                dose(units, seed, TemporalDoseTarget::SmoothedCurve { grid: dose_grid() })
            }
            Self::DoseGrid => dose(
                units,
                seed,
                TemporalDoseTarget::LocalQuadratic {
                    grid: dose_grid(),
                    derivative: false,
                    curvature_bound: Some(0.0),
                },
            ),
            Self::Quantile => quantile(units, seed),
        }
    }
}
// ------------------------------------------------------------------ panels

/// A two-arm micro-randomized panel: the treatment probability depends on the previous outcome,
/// and the outcome one period after a decision carries [`TWO_ARM_EFFECT`] of that decision and
/// `0.3` of the decision before it. The excursion effect of a decision on the next outcome is
/// [`TWO_ARM_EFFECT`].
fn two_arm(units: usize, seed: u64) -> (TabularData, TemporalDesignQuery) {
    let mut next = uniform(seed, 0x7D01);
    let (mut unit_labels, mut periods, mut assignment, mut probability, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..units {
        let level = next() - 0.5;
        let (mut previous_y, mut previous_a, mut before) = (0.0_f64, 0.0_f64, 0.0_f64);
        for t in 0..10_i64 {
            let p = if previous_y > 0.0 { 0.65 } else { 0.35 };
            let a = next() < p;
            let y = level + TWO_ARM_EFFECT * previous_a + 0.3 * before + (next() - 0.5);
            unit_labels.push(format!("u{i}"));
            periods.push(t);
            assignment.push(a);
            probability.push(p);
            outcome.push(y);
            previous_y = y;
            before = previous_a;
            previous_a = f64::from(u8::from(a));
        }
    }
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&unit_labels),
        periods: periods.as_slice().into(),
        lag: 1,
        assignment: TemporalAssignment::TwoArm {
            assignment: assignment.as_slice().into(),
            probability: probability.as_slice().into(),
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.05,
        outcome_misclassification: None,
    };
    (table(&[("y", outcome.as_slice())]), query)
}

/// Arm effects `0, 0.5, 1.5` of the multi-arm panel.
const MULTI_ARM_EFFECTS: [f64; 3] = [0.0, 0.5, 1.5];

/// A three-arm panel at lag 0 whose probability row moves with the period.
fn multi_arm(units: usize, seed: u64, categorical: bool) -> (TabularData, TemporalDesignQuery) {
    multi_arm_with(units, seed, categorical, MULTI_ARM_EFFECTS)
}

/// [`multi_arm`] with the arm effects `effects`.
fn multi_arm_with(
    units: usize,
    seed: u64,
    categorical: bool,
    effects: [f64; 3],
) -> (TabularData, TemporalDesignQuery) {
    let mut next = uniform(seed, 0x7D02);
    let (mut unit_labels, mut periods, mut assignment, mut probabilities, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..units {
        let level = next() - 0.5;
        for t in 0..6_i64 {
            let row = if t % 2 == 0 { vec![0.5, 0.25, 0.25] } else { vec![0.2, 0.4, 0.4] };
            let u = next();
            let arm = if u < row[0] {
                0_usize
            } else if u < row[0] + row[1] {
                1
            } else {
                2
            };
            unit_labels.push(format!("u{i}"));
            periods.push(t);
            assignment.push(arm);
            probabilities.push(row);
            outcome.push(level + effects[arm] + (next() - 0.5));
        }
    }
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&unit_labels),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::MultiArm {
            assignment: assignment.as_slice().into(),
            probabilities: probabilities.into(),
            arms: labels(&["low".into(), "mid".into(), "high".into()]),
            categorical,
            ordered: categorical,
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.1,
        outcome_misclassification: None,
    };
    (table(&[("y", outcome.as_slice())]), query)
}

/// Two binary factors assigned independently at every decision; the outcome adds `1.0` for the
/// first factor, `0.5` for the second and `0.4` for both.
fn factorial(units: usize, seed: u64) -> (TabularData, TemporalDesignQuery) {
    factorial_with(units, seed, FACTORIAL_EFFECTS)
}

/// First-factor, second-factor and joint additions of the factorial panel.
const FACTORIAL_EFFECTS: [f64; 3] = [1.0, 0.5, 0.4];

/// [`factorial`] with the additions `effects` (first factor, second factor, both).
fn factorial_with(
    units: usize,
    seed: u64,
    effects: [f64; 3],
) -> (TabularData, TemporalDesignQuery) {
    let mut next = uniform(seed, 0x7D03);
    let (mut unit_labels, mut periods, mut first, mut second, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..units {
        for t in 0..5_i64 {
            let a = next() < 0.5;
            let b = next() < 0.4;
            let (fa, fb) = (f64::from(u8::from(a)), f64::from(u8::from(b)));
            unit_labels.push(format!("u{i}"));
            periods.push(t);
            first.push(a);
            second.push(b);
            outcome.push(effects[0] * fa + effects[1] * fb + effects[2] * fa * fb + (next() - 0.5));
        }
    }
    let rows = unit_labels.len();
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&unit_labels),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::Factorial {
            first: first.as_slice().into(),
            first_probability: vec![0.5; rows].into(),
            second: second.as_slice().into(),
            second_probability: vec![0.4; rows].into(),
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.05,
        outcome_misclassification: None,
    };
    (table(&[("y", outcome.as_slice())]), query)
}

/// Pairs of units share an interference cluster; a unit's own treatment adds `1.0` and a
/// treated peer adds `0.6`, so the direct effect is `1.0`.
fn interference(clusters: usize, seed: u64) -> (TabularData, TemporalDesignQuery) {
    interference_with(clusters, seed, INTERFERENCE_EFFECTS)
}

/// Additions of a unit's own treatment and of a treated peer in the interference panel.
const INTERFERENCE_EFFECTS: [f64; 2] = [1.0, 0.6];

/// [`interference`] with the own-treatment and treated-peer additions `effects`.
fn interference_with(
    clusters: usize,
    seed: u64,
    effects: [f64; 2],
) -> (TabularData, TemporalDesignQuery) {
    let mut next = uniform(seed, 0x7D04);
    let (mut unit_labels, mut periods, mut cluster_labels, mut assignment) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for c in 0..clusters {
        for t in 0..4_i64 {
            for member in 0..2 {
                unit_labels.push(format!("u{c}_{member}"));
                periods.push(t);
                cluster_labels.push(format!("c{c}"));
                assignment.push(next() < 0.5);
            }
        }
    }
    let rows = unit_labels.len();
    let outcome: Vec<f64> = (0..rows)
        .map(|row| {
            let peer = if row % 2 == 0 { row + 1 } else { row - 1 };
            effects[0] * f64::from(u8::from(assignment[row]))
                + effects[1] * f64::from(u8::from(assignment[peer]))
                + (next() - 0.5)
        })
        .collect();
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&unit_labels),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::Interference {
            assignment: assignment.as_slice().into(),
            probability: vec![0.5; rows].into(),
            clusters: labels(&cluster_labels),
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.05,
        outcome_misclassification: None,
    };
    (table(&[("y", outcome.as_slice())]), query)
}

fn dose_grid() -> Arc<[f64]> {
    Arc::from([0.5, 1.0, 1.5])
}

/// The dose response `m(d)`.
fn dose_response(d: f64) -> f64 {
    1.0 + d + 0.5 * d * d
}

/// `int_0^DOSE_TOP K_h(u - at) m(u) du` by Simpson's rule: the expectation of the
/// inverse-density kernel score under the uniform dose law, which is the smoothed target.
fn smoothed_response(at: f64) -> f64 {
    let normal = BANDWIDTH * (2.0 * std::f64::consts::PI).sqrt();
    let integrand = |u: f64| {
        let z = (u - at) / BANDWIDTH;
        (-0.5 * z * z).exp() / normal * dose_response(u)
    };
    let steps = 4000_u32;
    let h = DOSE_TOP / f64::from(steps);
    let mut sum = integrand(0.0) + integrand(DOSE_TOP);
    for k in 1..steps {
        sum += integrand(h * f64::from(k)) * if k % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

/// Doses uniform on `(0, DOSE_TOP)` (density `1 / DOSE_TOP`) at every decision; the outcome is
/// `m(dose)` plus noise in the same period.
fn dose(units: usize, seed: u64, target: TemporalDoseTarget) -> (TabularData, TemporalDesignQuery) {
    dose_with(units, seed, target, dose_response)
}

/// [`dose`] with the dose response `response`.
fn dose_with(
    units: usize,
    seed: u64,
    target: TemporalDoseTarget,
    response: fn(f64) -> f64,
) -> (TabularData, TemporalDesignQuery) {
    let mut next = uniform(seed, 0x7D05);
    let (mut unit_labels, mut periods, mut doses, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..units {
        for t in 0..8_i64 {
            let d = DOSE_TOP * next();
            unit_labels.push(format!("u{i}"));
            periods.push(t);
            doses.push(d);
            outcome.push(response(d) + (next() - 0.5));
        }
    }
    let rows = unit_labels.len();
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&unit_labels),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::Dose {
            dose: doses.as_slice().into(),
            density: vec![1.0 / DOSE_TOP; rows].into(),
            bandwidth: BANDWIDTH,
            target,
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.05,
        outcome_misclassification: None,
    };
    (table(&[("y", outcome.as_slice())]), query)
}

/// A two-arm panel whose outcome is a uniform draw plus `2` under treatment, so the median of
/// the outcome under either arm is `0.5` and `2.5` and the median shift is exactly `2`.
fn quantile(units: usize, seed: u64) -> (TabularData, TemporalDesignQuery) {
    quantile_at(units, seed, 0.5)
}

/// [`quantile`] at the quantile level `level`: the arm quantiles are `level` and `level + 2`,
/// so the shift is `2` at every level.
fn quantile_at(units: usize, seed: u64, level: f64) -> (TabularData, TemporalDesignQuery) {
    let mut next = uniform(seed, 0x7D06);
    let (mut unit_labels, mut periods, mut assignment, mut probability, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..units {
        for t in 0..6_i64 {
            let p = if t % 2 == 0 { 0.4 } else { 0.6 };
            let a = next() < p;
            unit_labels.push(format!("u{i}"));
            periods.push(t);
            assignment.push(a);
            probability.push(p);
            outcome.push(next() + if a { 2.0 } else { 0.0 });
        }
    }
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&unit_labels),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::TwoArm {
            assignment: assignment.as_slice().into(),
            probability: probability.as_slice().into(),
        },
        functional: TemporalDesignFunctional::Quantile { level },
        min_probability: 0.05,
        outcome_misclassification: None,
    };
    (table(&[("y", outcome.as_slice())]), query)
}

/// Confounder at period `k`: `L_k = 0.5 L_{k-1} - 0.4 A_{k-1} + (u - 0.5)`.
fn next_confounder(current: f64, previous_treated: bool, noise: f64) -> f64 {
    0.5 * current - 0.4 * f64::from(u8::from(previous_treated)) + (noise - 0.5)
}

/// Subject trajectories with time-varying confounding, followed for `horizon` periods: treatment
/// is more likely when the confounder is high, the hazard is `expit(-2.5 + 0.8 L - 0.7 A)` and
/// censoring is light.
fn survival_at(subjects: usize, seed: u64, horizon: u32) -> (TabularData, TemporalDesignQuery) {
    let mut next = uniform(seed, 0x7D07);
    let (mut unit_labels, mut periods) = (Vec::new(), Vec::new());
    let (mut events, mut treated, mut censored, mut confounder) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..subjects {
        let mut current = 2.0 * next() - 1.0;
        let mut previous = false;
        for k in 0..horizon {
            if k > 0 {
                current = next_confounder(current, previous, next());
            }
            let a = next() < expit(0.8 * current);
            let c = next() < 0.03;
            let y = !c && next() < expit(-2.5 + 0.8 * current - 0.7 * f64::from(u8::from(a)));
            unit_labels.push(format!("s{i}"));
            periods.push(i64::from(k));
            events.push(f64::from(u8::from(y)));
            treated.push(f64::from(u8::from(a)));
            censored.push(f64::from(u8::from(c)));
            confounder.push(current);
            previous = a;
            if c || y {
                break;
            }
        }
    }
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&unit_labels),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::RegimeSurvival {
            treatment: VariableId::from_raw(1),
            censored: VariableId::from_raw(2),
            confounders: Arc::from([VariableId::from_raw(3)]),
            horizon,
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.02,
        outcome_misclassification: None,
    };
    (
        table(&[
            ("event", events.as_slice()),
            ("treated", treated.as_slice()),
            ("censored", censored.as_slice()),
            ("confounder", confounder.as_slice()),
        ]),
        query,
    )
}

/// Always-treat minus never-treat survival at `horizon` periods, by simulating the generating
/// law under each static regime (the hazards are multiplied out, so only the confounder path is
/// random).
fn simulated_survival_difference(horizon: u32) -> f64 {
    let draws = 2_000_000_u32;
    let mut next = uniform(0x5A17_0001, 0x7D08);
    let mut totals = [0.0_f64; 2];
    for (index, arm) in [true, false].into_iter().enumerate() {
        for _ in 0..draws {
            let mut current = 2.0 * next() - 1.0;
            let mut surviving = 1.0;
            for k in 0..horizon {
                if k > 0 {
                    current = next_confounder(current, arm, next());
                }
                surviving *= 1.0 - expit(-2.5 + 0.8 * current - 0.7 * f64::from(u8::from(arm)));
            }
            totals[index] += surviving;
        }
    }
    (totals[0] - totals[1]) / f64::from(draws)
}
// --------------------------------------------------------------- execution

// ------------------------------------------- further published statements

/// Family-wise level of the Holm family: no adjusted p-value below it is a non-rejection.
const HOLM_ALPHA: f64 = 0.05;

/// Constant third derivative of [`cubic_response`], declared exactly as its curvature bound.
const CUBIC_CURVATURE: f64 = 3.0;

/// Horizon of the longer survival panel.
const LONGER_HORIZON: u32 = 4;

/// A dose response whose third derivative is [`CUBIC_CURVATURE`] everywhere.
fn cubic_response(d: f64) -> f64 {
    dose_response(d) + CUBIC_CURVATURE / 6.0 * (d - 1.0).powi(3)
}

/// The carryover inside both arm means of the two-arm panel. A lag-one pair at decision `t`
/// has `E[Y_{t+1}(a)] = 1.25 a + 0.3 E[A_{t-1}]` (the unit level has mean zero and `A_{-1} = 0`),
/// so each arm mean over the nine pairs adds `0.3 * sum_{s=0}^{7} E[A_s] / 9`; the history-
/// dependent `E[A_s]` is simulated from the generating law.
fn two_arm_carryover() -> f64 {
    static TRUTH: OnceLock<f64> = OnceLock::new();
    *TRUTH.get_or_init(|| {
        let draws = 2_000_000_u32;
        let mut next = uniform(0x5A17_0002, 0x7D09);
        let mut treated = 0.0_f64;
        for _ in 0..draws {
            let level = next() - 0.5;
            let (mut previous_y, mut previous_a, mut before) = (0.0_f64, 0.0_f64, 0.0_f64);
            for _ in 0..8 {
                let p = if previous_y > 0.0 { 0.65 } else { 0.35 };
                let a = f64::from(u8::from(next() < p));
                previous_y = level + TWO_ARM_EFFECT * previous_a + 0.3 * before + (next() - 0.5);
                before = previous_a;
                previous_a = a;
                treated += a;
            }
        }
        0.3 * treated / (9.0 * f64::from(draws))
    })
}

/// [`simulated_survival_difference`] at [`LONGER_HORIZON`].
fn longer_survival_truth() -> f64 {
    static TRUTH: OnceLock<f64> = OnceLock::new();
    *TRUTH.get_or_init(|| simulated_survival_difference(LONGER_HORIZON))
}

/// The simultaneous band of one fit.
struct Band {
    coordinates: Vec<usize>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

/// Everything one frequentist fit publishes, with the construction the runtime reports.
struct Published {
    coordinates: Vec<String>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    band: Option<Band>,
    holm: Option<Vec<f64>>,
    construction: Construction,
    scope: ScopeFacts,
}

impl Published {
    /// The runtime's construction with the declared statement beside it, as [`execute`] keys a
    /// frequentist record.
    fn declared(&self, method: &str, se_kind: &str) -> Construction {
        let mut construction = self.construction.clone();
        construction.interval_method = method.into();
        construction.se_kind = se_kind.into();
        construction
    }

    /// The pointwise interval of the coordinate named `name`.
    fn interval(&self, name: &str) -> Option<(f64, f64)> {
        let index = self.coordinates.iter().position(|coordinate| coordinate == name)?;
        Some((*self.lower.get(index)?, *self.upper.get(index)?))
    }

    /// The band envelope and truth of one coordinate that decides joint coverage: the first
    /// coordinate the band misses, or, when it covers every one, the coordinate whose truth
    /// sits closest to an edge. `None` when the band is absent or is not over `targets`.
    fn deciding_band_coordinate(&self, targets: &[(&str, f64)]) -> Option<((f64, f64), f64)> {
        let band = self.band.as_ref()?;
        if band.coordinates.len() != targets.len() {
            return None;
        }
        let mut closest: Option<(f64, (f64, f64), f64)> = None;
        for (slot, &index) in band.coordinates.iter().enumerate() {
            let name = self.coordinates.get(index)?;
            let truth = targets.iter().find(|(target, _)| target == name)?.1;
            let (lower, upper) = (*band.lower.get(slot)?, *band.upper.get(slot)?);
            if !(lower <= truth && truth <= upper) {
                return Some(((lower, upper), truth));
            }
            let margin = (truth - lower).min(upper - truth) / (upper - lower);
            if closest.is_none_or(|(least, _, _)| margin < least) {
                closest = Some((margin, (lower, upper), truth));
            }
        }
        closest.map(|(_, envelope, truth)| (envelope, truth))
    }
}

/// Run a frequentist study of one panel; `None` when it is refused.
fn publish(
    (data, query): (TabularData, TemporalDesignQuery),
    replicates: u32,
    seed: u64,
) -> Option<Published> {
    let ctx = ExecutionContext::for_tests(seed);
    let mut builder = Study::tabular(data.clone())
        .query(CausalQuery::TemporalDesign(query))
        .inference(InferenceMode::Frequentist);
    if replicates > 0 {
        builder = builder.bootstrap_replicates(replicates);
    }
    let study = builder.build().ok()?;
    let prepared = study.prepare(&ctx).ok()?;
    let result = prepared.estimate(&data, &ctx).ok()?;
    let fit = result.temporal_design()?;
    let contract = prepared.contract_for_result(&result).ok()?;
    let (mut construction, mut scope, _) = constructions(&contract, &result).into_iter().next()?;
    construction.structure = "design".into();
    construction.dependence = "cluster".into();
    construction.reported_level = REPORTED_LEVEL;
    scope.replicates_ok = fit.replicates.and_then(|(ok, _)| u32::try_from(ok).ok());
    let band = match (&fit.band_lower, &fit.band_upper) {
        (Some(lower), Some(upper)) if !fit.band_coordinates.is_empty() => Some(Band {
            coordinates: fit.band_coordinates.to_vec(),
            lower: lower.to_vec(),
            upper: upper.to_vec(),
        }),
        _ => None,
    };
    Some(Published {
        coordinates: fit.coordinates.iter().map(ToString::to_string).collect(),
        lower: fit.lower.to_vec(),
        upper: fit.upper.to_vec(),
        band,
        holm: fit.holm_adjusted_p.as_ref().map(|values| values.to_vec()),
        construction,
        scope,
    })
}

/// One frequentist measurement: where its panels come from and what its records are keyed on.
#[derive(Clone, Copy)]
struct Measurement {
    test: &'static str,
    dgp: &'static str,
    units: usize,
    seed_base: u64,
    /// Whole-procedure bootstrap replicates; zero for an analytic interval.
    replicates: u32,
}

impl Measurement {
    /// Fit `n_sim()` panels drawn by `draw`.
    fn fits(
        self,
        draw: impl Fn(usize, u64) -> (TabularData, TemporalDesignQuery) + Sync,
    ) -> Vec<Option<Published>> {
        map_replicates(n_sim(), |rep| {
            let seed = self.seed_base + rep;
            publish(draw(self.units, seed), self.replicates, seed)
        })
    }

    /// One tally per named coordinate: its pointwise interval against its truth.
    fn pointwise(
        self,
        runs: &[Option<Published>],
        (method, se_kind): (&'static str, &'static str),
        targets: &[(&'static str, f64)],
    ) -> Vec<CoverageTally> {
        let key = RecordKey { test: self.test, dgp: self.dgp, interval: method };
        targets
            .iter()
            .map(|&(name, truth)| {
                let mut tally = CoverageTally::for_record(key, REPORTED_LEVEL).labelled(name);
                for run in runs {
                    let Some(run) = run else {
                        tally.skip();
                        continue;
                    };
                    tally.bind_validation(
                        &run.declared(method, se_kind),
                        run.scope,
                        "not_applicable",
                    );
                    tally.record(run.interval(name), truth);
                }
                tally
            })
            .collect()
    }

    /// Joint coverage of the simultaneous band over exactly `targets`: a replicate covers when
    /// every coordinate's truth is inside its envelope. A fit without that band is a miss.
    fn band(
        self,
        runs: &[Option<Published>],
        se_kind: &'static str,
        targets: &[(&'static str, f64)],
    ) -> CoverageTally {
        let key = RecordKey { test: self.test, dgp: self.dgp, interval: "simultaneous_band" };
        let mut tally = CoverageTally::for_record(key, REPORTED_LEVEL).labelled("band");
        for run in runs {
            let Some(run) = run else {
                tally.skip();
                continue;
            };
            tally.bind_validation(
                &run.declared("simultaneous_band", se_kind),
                run.scope,
                "not_applicable",
            );
            match run.deciding_band_coordinate(targets) {
                Some((envelope, truth)) => tally.record(Some(envelope), truth),
                None => tally.record(None, targets[0].1),
            }
        }
        tally
    }

    /// Family-wise error of the Holm-adjusted p-values on panels whose every family contrast
    /// is zero: a replicate "covers" the null when no adjusted p-value is below
    /// [`HOLM_ALPHA`]. A fit without the `family` p-values is a miss.
    fn holm(self, runs: &[Option<Published>], family: usize) -> CoverageTally {
        let key = RecordKey { test: self.test, dgp: self.dgp, interval: "holm_adjusted_p" };
        let mut tally = CoverageTally::for_record(key, REPORTED_LEVEL);
        for run in runs {
            let Some(run) = run else {
                tally.skip();
                continue;
            };
            tally.bind_validation(
                &run.declared("holm_adjusted_p", "cluster_ratio"),
                run.scope,
                "not_applicable",
            );
            let smallest = run
                .holm
                .as_ref()
                .filter(|values| values.len() == family && values.iter().all(|p| p.is_finite()))
                .map(|values| values.iter().copied().fold(f64::INFINITY, f64::min));
            tally.absorb(&smallest);
            tally.record(
                smallest.map(|p| if p < HOLM_ALPHA { (1.0, 2.0) } else { (-1.0, 1.0) }),
                0.0,
            );
        }
        tally
    }

    /// Assert every tally, each on its own record, and report every failure together.
    /// `conservative` names the target of a statement that is at least nominal by
    /// construction; such a tally is held to the one-sided floor.
    fn conclude(self, tallies: &[CoverageTally], conservative: Option<&'static str>) {
        let failures: Vec<String> = tallies
            .iter()
            .filter_map(|tally| {
                std::panic::catch_unwind(|| match conservative {
                    Some(target) => tally.assert_one_sided(CoverageSide::Upper, target),
                    None => tally.assert(),
                })
                .err()
                .map(|error| {
                    error
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| error.downcast_ref::<&str>().map(|text| (*text).to_owned()))
                        .unwrap_or_else(|| "coverage failure".into())
                })
            })
            .collect();
        assert!(failures.is_empty(), "{}: {}", self.test, failures.join("; "));
    }
}

/// The declared frequentist interval of the arm and smoothed-dose designs.
const CLUSTER_RATIO: (&str, &str) = ("analytic_se", "cluster_ratio");

/// The declared frequentist interval of the local-quadratic dose grid.
const BIAS_AWARE: (&str, &str) = ("analytic_se", "local_quadratic_bias_aware");

/// The declared frequentist interval of the bootstrap designs.
const PERCENTILE: (&str, &str) = ("percentile_bootstrap", "percentile");

/// Standard-error kind declared for a Gaussian multiplier sup-t band over clusters.
const SUP_T: &str = "sup_t_cluster_multiplier";

/// Standard-error kind declared for the sup-t band widened by the smoothing-bias bound.
const SUP_T_BIAS_AWARE: &str = "sup_t_local_quadratic_bias_aware";

/// Arm means of the multi-arm panel: the unit level and the noise have mean zero.
const MULTI_ARM_MEANS: [(&str, f64); 3] = [
    ("mean:low", MULTI_ARM_EFFECTS[0]),
    ("mean:mid", MULTI_ARM_EFFECTS[1]),
    ("mean:high", MULTI_ARM_EFFECTS[2]),
];

/// Contrasts against the reference arm, the band and Holm family of the multi-arm panel.
const MULTI_ARM_CONTRASTS: [(&str, f64); 2] = [
    ("contrast:mid-low", MULTI_ARM_EFFECTS[1] - MULTI_ARM_EFFECTS[0]),
    ("contrast:high-low", MULTI_ARM_EFFECTS[2] - MULTI_ARM_EFFECTS[0]),
];

/// Main effects (each averaged over the other factor with equal weight) and the interaction.
fn factorial_effects() -> [(&'static str, f64); 3] {
    let [first, second, both] = FACTORIAL_EFFECTS;
    [
        ("main_effect:first", first + 0.5 * both),
        ("main_effect:second", second + 0.5 * both),
        ("interaction", both),
    ]
}

/// Direct, spillover and total effects against the untreated, unexposed mean of zero.
fn interference_effects() -> [(&'static str, f64); 3] {
    let [own, peer] = INTERFERENCE_EFFECTS;
    [("direct_effect", own), ("spillover_effect", peer), ("total_effect", own + peer)]
}

/// The dose grid's coordinates named `label@dose` with the value of `truth` at each dose.
fn grid_targets(label: &str, truth: fn(f64) -> f64) -> Vec<(&'static str, f64)> {
    dose_grid()
        .iter()
        .map(|&at| {
            let name: &'static str = Box::leak(format!("{label}@{at}").into_boxed_str());
            (name, truth(at))
        })
        .collect()
}

/// A local-quadratic dose panel with response `response` and the declared curvature bound.
fn local_quadratic(
    derivative: bool,
    bound: f64,
    response: fn(f64) -> f64,
) -> impl Fn(usize, u64) -> (TabularData, TemporalDesignQuery) + Sync {
    move |units, seed| {
        dose_with(
            units,
            seed,
            TemporalDoseTarget::LocalQuadratic {
                grid: dose_grid(),
                derivative,
                curvature_bound: Some(bound),
            },
            response,
        )
    }
}
// The other coordinates of each design, on the panels of the first group.

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_two_arm_arm_mean_intervals() {
    let measurement = Measurement {
        test: "temporal_design_two_arm_arm_mean_intervals",
        dgp: "sequential_two_arm_history_dependent_lag_one_known_excursion",
        units: Design::TwoArm.units(),
        seed_base: 0x7D10_0000,
        replicates: 0,
    };
    let runs = measurement.fits(two_arm);
    let carryover = two_arm_carryover();
    let targets = [("treated_mean", TWO_ARM_EFFECT + carryover), ("control_mean", carryover)];
    measurement.conclude(&measurement.pointwise(&runs, CLUSTER_RATIO, &targets), None);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_multi_arm_other_coordinate_intervals() {
    let measurement = Measurement {
        test: "temporal_design_multi_arm_other_coordinate_intervals",
        dgp: "sequential_three_arm_period_varying_probabilities_known_contrast",
        units: Design::MultiArm.units(),
        seed_base: 0x7D11_0000,
        replicates: 0,
    };
    let runs = measurement.fits(|units, seed| multi_arm(units, seed, false));
    let mut targets = vec![MULTI_ARM_CONTRASTS[1]];
    targets.extend(MULTI_ARM_MEANS);
    measurement.conclude(&measurement.pointwise(&runs, CLUSTER_RATIO, &targets), None);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_categorical_other_coordinate_intervals() {
    let measurement = Measurement {
        test: "temporal_design_categorical_other_coordinate_intervals",
        dgp: "sequential_three_level_period_varying_probabilities_known_ordered_step",
        units: Design::Categorical.units(),
        seed_base: 0x7D12_0000,
        replicates: 0,
    };
    let runs = measurement.fits(|units, seed| multi_arm(units, seed, true));
    let mut targets = MULTI_ARM_CONTRASTS.to_vec();
    targets.extend(MULTI_ARM_MEANS);
    measurement.conclude(&measurement.pointwise(&runs, CLUSTER_RATIO, &targets), None);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_factorial_other_coordinate_intervals() {
    let measurement = Measurement {
        test: "temporal_design_factorial_other_coordinate_intervals",
        dgp: "sequential_two_factor_independent_assignment_known_main_effect",
        units: Design::Factorial.units(),
        seed_base: 0x7D13_0000,
        replicates: 0,
    };
    let runs = measurement.fits(factorial);
    let [first, second, both] = FACTORIAL_EFFECTS;
    let mut targets = factorial_effects()[1..].to_vec();
    // Cell means in the order first-factor level, second-factor level.
    targets.extend([
        ("mean:cell:00", 0.0),
        ("mean:cell:10", first),
        ("mean:cell:01", second),
        ("mean:cell:11", first + second + both),
    ]);
    measurement.conclude(&measurement.pointwise(&runs, CLUSTER_RATIO, &targets), None);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_interference_other_coordinate_intervals() {
    let measurement = Measurement {
        test: "temporal_design_interference_other_coordinate_intervals",
        dgp: "sequential_paired_cluster_spillover_known_direct_effect",
        units: Design::Interference.units(),
        seed_base: 0x7D14_0000,
        replicates: 0,
    };
    let runs = measurement.fits(interference);
    let [own, peer] = INTERFERENCE_EFFECTS;
    let mut targets = interference_effects()[1..].to_vec();
    // Exposure means by own treatment and whether the peer is treated.
    targets.extend([
        ("mean:own0_peers0", 0.0),
        ("mean:own1_peers0", own),
        ("mean:own0_peers1", peer),
        ("mean:own1_peers1", own + peer),
    ]);
    measurement.conclude(&measurement.pointwise(&runs, CLUSTER_RATIO, &targets), None);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_continuous_dose_response_intervals() {
    let measurement = Measurement {
        test: "temporal_design_continuous_dose_response_intervals",
        dgp: "uniform_dose_quadratic_response_known_smoothed_contrast",
        units: Design::ContinuousDose.units(),
        seed_base: 0x7D15_0000,
        replicates: 0,
    };
    let runs = measurement.fits(|units, seed| Design::ContinuousDose.draw(units, seed));
    let targets = [
        ("smoothed_response@1.2", smoothed_response(1.2)),
        ("smoothed_response@0.8", smoothed_response(0.8)),
    ];
    measurement.conclude(&measurement.pointwise(&runs, CLUSTER_RATIO, &targets), None);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_smoothed_dose_other_grid_intervals() {
    let measurement = Measurement {
        test: "temporal_design_smoothed_dose_other_grid_intervals",
        dgp: "uniform_dose_quadratic_response_known_smoothed_response",
        units: Design::SmoothedDose.units(),
        seed_base: 0x7D16_0000,
        replicates: 0,
    };
    let runs = measurement.fits(|units, seed| Design::SmoothedDose.draw(units, seed));
    let targets = [
        ("smoothed_response@0.5", smoothed_response(0.5)),
        ("smoothed_response@1.5", smoothed_response(1.5)),
    ];
    measurement.conclude(&measurement.pointwise(&runs, CLUSTER_RATIO, &targets), None);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_dose_grid_other_level_intervals() {
    let measurement = Measurement {
        test: "temporal_design_dose_grid_other_level_intervals",
        dgp: "uniform_dose_quadratic_response_known_level_zero_curvature_bound",
        units: Design::DoseGrid.units(),
        seed_base: 0x7D17_0000,
        replicates: 0,
    };
    let runs = measurement.fits(|units, seed| Design::DoseGrid.draw(units, seed));
    let targets = [("level@0.5", dose_response(0.5)), ("level@1.5", dose_response(1.5))];
    measurement.conclude(&measurement.pointwise(&runs, BIAS_AWARE, &targets), None);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_quantile_arm_median_bootstrap_intervals() {
    let measurement = Measurement {
        test: "temporal_design_quantile_arm_median_bootstrap_intervals",
        dgp: "sequential_two_arm_location_shift_known_median_shift",
        units: Design::Quantile.units(),
        seed_base: 0x7D18_0000,
        replicates: BOOTSTRAP_REPLICATES,
    };
    let runs = measurement.fits(quantile);
    // The outcome is uniform on (0, 1), shifted by 2 under treatment.
    let targets = [("treated_quantile", 2.5), ("control_quantile", 0.5)];
    measurement.conclude(&measurement.pointwise(&runs, PERCENTILE, &targets), None);
}
// Simultaneous bands: joint coverage of every band coordinate.

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_multi_arm_contrast_band() {
    let measurement = Measurement {
        test: "temporal_design_multi_arm_contrast_band",
        dgp: "sequential_three_arm_period_varying_probabilities_known_contrast",
        units: Design::MultiArm.units(),
        seed_base: 0x7D11_0000,
        replicates: 0,
    };
    let runs = measurement.fits(|units, seed| multi_arm(units, seed, false));
    measurement.conclude(&[measurement.band(&runs, SUP_T, &MULTI_ARM_CONTRASTS)], None);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_categorical_contrast_band() {
    let measurement = Measurement {
        test: "temporal_design_categorical_contrast_band",
        dgp: "sequential_three_level_period_varying_probabilities_known_ordered_step",
        units: Design::Categorical.units(),
        seed_base: 0x7D12_0000,
        replicates: 0,
    };
    let runs = measurement.fits(|units, seed| multi_arm(units, seed, true));
    measurement.conclude(&[measurement.band(&runs, SUP_T, &MULTI_ARM_CONTRASTS)], None);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_factorial_effect_band() {
    let measurement = Measurement {
        test: "temporal_design_factorial_effect_band",
        dgp: "sequential_two_factor_independent_assignment_known_main_effect",
        units: Design::Factorial.units(),
        seed_base: 0x7D13_0000,
        replicates: 0,
    };
    let runs = measurement.fits(factorial);
    measurement.conclude(&[measurement.band(&runs, SUP_T, &factorial_effects())], None);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_interference_effect_band() {
    let measurement = Measurement {
        test: "temporal_design_interference_effect_band",
        dgp: "sequential_paired_cluster_spillover_known_direct_effect",
        units: Design::Interference.units(),
        seed_base: 0x7D14_0000,
        replicates: 0,
    };
    let runs = measurement.fits(interference);
    measurement.conclude(&[measurement.band(&runs, SUP_T, &interference_effects())], None);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_smoothed_dose_curve_band() {
    let measurement = Measurement {
        test: "temporal_design_smoothed_dose_curve_band",
        dgp: "uniform_dose_quadratic_response_known_smoothed_response",
        units: Design::SmoothedDose.units(),
        seed_base: 0x7D16_0000,
        replicates: 0,
    };
    let runs = measurement.fits(|units, seed| Design::SmoothedDose.draw(units, seed));
    let targets = grid_targets("smoothed_response", smoothed_response);
    measurement.conclude(&[measurement.band(&runs, SUP_T, &targets)], None);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_dose_grid_level_band() {
    let measurement = Measurement {
        test: "temporal_design_dose_grid_level_band",
        dgp: "uniform_dose_quadratic_response_known_level_zero_curvature_bound",
        units: Design::DoseGrid.units(),
        seed_base: 0x7D17_0000,
        replicates: 0,
    };
    let runs = measurement.fits(|units, seed| Design::DoseGrid.draw(units, seed));
    let targets = grid_targets("level", dose_response);
    measurement.conclude(&[measurement.band(&runs, SUP_T_BIAS_AWARE, &targets)], None);
}
// Holm-adjusted p-values: family-wise error on panels whose every family contrast is zero.
// Under that null Holm rejects exactly when the smallest raw p-value is below alpha over the
// family size, so its error is at most alpha by the Bonferroni inequality: the non-rejection
// rate is held to the one-sided floor.

/// What a Holm record covers.
const HOLM_TARGET: &str = "no family contrast rejected at 0.05 when every family contrast is zero";

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_multi_arm_holm_family_error() {
    let measurement = Measurement {
        test: "temporal_design_multi_arm_holm_family_error",
        dgp: "sequential_three_arm_period_varying_probabilities_no_arm_effect",
        units: Design::MultiArm.units(),
        seed_base: 0x7D20_0000,
        replicates: 0,
    };
    let runs = measurement.fits(|units, seed| multi_arm_with(units, seed, false, [0.0; 3]));
    measurement.conclude(&[measurement.holm(&runs, 2)], Some(HOLM_TARGET));
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_categorical_holm_family_error() {
    let measurement = Measurement {
        test: "temporal_design_categorical_holm_family_error",
        dgp: "sequential_three_level_period_varying_probabilities_no_level_effect",
        units: Design::Categorical.units(),
        seed_base: 0x7D21_0000,
        replicates: 0,
    };
    let runs = measurement.fits(|units, seed| multi_arm_with(units, seed, true, [0.0; 3]));
    measurement.conclude(&[measurement.holm(&runs, 2)], Some(HOLM_TARGET));
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_factorial_holm_family_error() {
    let measurement = Measurement {
        test: "temporal_design_factorial_holm_family_error",
        dgp: "sequential_two_factor_independent_assignment_no_factor_effect",
        units: Design::Factorial.units(),
        seed_base: 0x7D22_0000,
        replicates: 0,
    };
    let runs = measurement.fits(|units, seed| factorial_with(units, seed, [0.0; 3]));
    measurement.conclude(&[measurement.holm(&runs, 3)], Some(HOLM_TARGET));
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_interference_holm_family_error() {
    let measurement = Measurement {
        test: "temporal_design_interference_holm_family_error",
        dgp: "sequential_paired_cluster_no_direct_or_spillover_effect",
        units: Design::Interference.units(),
        seed_base: 0x7D23_0000,
        replicates: 0,
    };
    let runs = measurement.fits(|units, seed| interference_with(units, seed, [0.0; 2]));
    measurement.conclude(&[measurement.holm(&runs, 3)], Some(HOLM_TARGET));
}
// Quantile levels beside the median. The outcome is uniform on (0, 1) plus 2 under treatment,
// so the arm quantiles at level q are q and q + 2 and the shift is 2.

fn measure_quantile_level(test: &'static str, dgp: &'static str, level: f64, seed_base: u64) {
    let measurement = Measurement {
        test,
        dgp,
        units: Design::Quantile.units(),
        seed_base,
        replicates: BOOTSTRAP_REPLICATES,
    };
    let runs = measurement.fits(|units, seed| quantile_at(units, seed, level));
    let targets =
        [("quantile_effect", 2.0), ("treated_quantile", level + 2.0), ("control_quantile", level)];
    measurement.conclude(&measurement.pointwise(&runs, PERCENTILE, &targets), None);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_lower_quartile_effect_bootstrap_intervals() {
    measure_quantile_level(
        "temporal_design_lower_quartile_effect_bootstrap_intervals",
        "sequential_two_arm_location_shift_known_lower_quartile_shift",
        0.25,
        0x7D24_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_upper_quartile_effect_bootstrap_intervals() {
    measure_quantile_level(
        "temporal_design_upper_quartile_effect_bootstrap_intervals",
        "sequential_two_arm_location_shift_known_upper_quartile_shift",
        0.75,
        0x7D25_0000,
    );
}
// The local-quadratic dose grid: the derivative, and the bias-aware interval and band under a
// non-zero declared curvature bound.

/// Pointwise intervals and the band of one local-quadratic grid, on the same fits.
fn measure_local_quadratic(
    measurement: Measurement,
    draw: impl Fn(usize, u64) -> (TabularData, TemporalDesignQuery) + Sync,
    targets: &[(&'static str, f64)],
    conservative: Option<&'static str>,
) {
    let runs = measurement.fits(draw);
    let mut tallies = measurement.pointwise(&runs, BIAS_AWARE, targets);
    tallies.push(measurement.band(&runs, SUP_T_BIAS_AWARE, targets));
    measurement.conclude(&tallies, conservative);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_dose_grid_derivative_intervals() {
    // The curve is quadratic, which a local quadratic reproduces: the slope is unbiased for
    // m'(d) = 1 + d and the zero curvature bound is exact.
    measure_local_quadratic(
        Measurement {
            test: "temporal_design_dose_grid_derivative_intervals",
            dgp: "uniform_dose_quadratic_response_known_derivative_zero_curvature_bound",
            units: Design::DoseGrid.units(),
            seed_base: 0x7D26_0000,
            replicates: 0,
        },
        local_quadratic(true, 0.0, dose_response),
        &grid_targets("derivative", |d: f64| 1.0 + d),
        None,
    );
}

/// What a bias-aware record covers. The interval and band widen by the worst smoothing bias
/// over every curve within the declared bound, so they are at least nominal for any one curve.
const BIAS_AWARE_TARGET: &str =
    "the dose response covered at least nominally under the declared third-derivative bound";

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_dose_grid_bias_aware_level_intervals() {
    measure_local_quadratic(
        Measurement {
            test: "temporal_design_dose_grid_bias_aware_level_intervals",
            dgp: "uniform_dose_cubic_response_known_level_exact_curvature_bound",
            units: Design::DoseGrid.units(),
            seed_base: 0x7D27_0000,
            replicates: 0,
        },
        local_quadratic(false, CUBIC_CURVATURE, cubic_response),
        &grid_targets("level", cubic_response),
        Some(BIAS_AWARE_TARGET),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_dose_grid_bias_aware_derivative_intervals() {
    // m'(d) = 1 + d + (CUBIC_CURVATURE / 2) (d - 1)^2 for the cubic response.
    measure_local_quadratic(
        Measurement {
            test: "temporal_design_dose_grid_bias_aware_derivative_intervals",
            dgp: "uniform_dose_cubic_response_known_derivative_exact_curvature_bound",
            units: Design::DoseGrid.units(),
            seed_base: 0x7D28_0000,
            replicates: 0,
        },
        local_quadratic(true, CUBIC_CURVATURE, cubic_response),
        &grid_targets("derivative", |d: f64| 1.0 + d + 0.5 * CUBIC_CURVATURE * (d - 1.0).powi(2)),
        Some(BIAS_AWARE_TARGET),
    );
}
// Regime survival past three periods, on the generating law of the first group.

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_regime_survival_longer_horizon_bootstrap_interval() {
    let measurement = Measurement {
        test: "temporal_design_regime_survival_longer_horizon_bootstrap_interval",
        dgp: "time_varying_confounding_logistic_hazard_known_four_period_survival_difference",
        // Twice the subjects of the three-period panel: about half as many follow a static
        // regime through each further period.
        units: grid_n(2400),
        seed_base: 0x7D29_0000,
        replicates: BOOTSTRAP_REPLICATES,
    };
    let runs = measurement.fits(|subjects, seed| survival_at(subjects, seed, LONGER_HORIZON));
    let targets = [("survival_difference", longer_survival_truth())];
    measurement.conclude(&measurement.pointwise(&runs, PERCENTILE, &targets), None);
}

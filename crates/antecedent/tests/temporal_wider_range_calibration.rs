//! Coverage of the temporal-design, sequential-contrast and regime-survival intervals over
//! ranges their support rows state and the base designs do not measure.
//!
//! Every design repeats a base design (`temporal_design_calibration.rs`,
//! `temporal_routes_calibration.rs`) with one axis moved and nothing else changed. The designs
//! were fixed before any run and are not tuned.
//!
//! Decision lag. The temporal-design rows pair each decision with the same unit's outcome `lag`
//! periods later and state no bound on the lag; the base designs measure lag one (two arms) and
//! lag zero (the other designs). Every design kind whose route accepts a lag is repeated at lag
//! two and at lag three: the outcome at period `t` is the base outcome with the decision of
//! period `t - lag` in the place the base design reads the decision of period `t` (lag zero) or
//! `t - 1` (two arms, whose carryover term moves with it to `t - lag - 1`). Each panel spans
//! `lag` more periods than the base so the number of decision-outcome pairs per unit, and so
//! the support the row counts, is the base design's. The truth is the base design's, since the
//! lag moves where the effect lands and not its size. Regime survival refuses a nonzero lag
//! (the query's validation), so it has no lag design.
//!
//! Probability floors. The assignment probabilities sit on the row's floor at every decision:
//! two arms treated with probability `0.35` (floor `0.35`); the interference clusters treated
//! with probability `0.25` (floor `0.25`); the three-arm and three-level designs alternating
//! the rows `(0.6, 0.2, 0.2)` and `(0.2, 0.2, 0.6)` (floor `0.2`, the middle arm at the floor in
//! every period); the factorial treatment probabilities `0.4` and `0.5` (joint cell floor `0.2`); the
//! quantile design treated with probability `0.4` (floor `0.4`).
//!
//! Wide fitted probabilities of the sequential contrast. The base panel keeps every action
//! probability in `[.25, .75]`; this one keeps the same outcome model and widens the assignment
//! models, `a0 ~ Bernoulli(expit(1.5 l0))` and `a1 ~ Bernoulli(expit(1.2 (l1 - .4)))`, so the
//! true probabilities reach `0.069` and `0.931` (first decision) and `0.072` and `0.928`
//! (second). The truth is the base design's, `1.32 + 1.5`.
//!
//! Regime survival. The base design has true regime-consistent treatment probabilities above
//! `0.2` and about three percent censoring per period. Low overlap treats with probability
//! `expit(1.9 (L + 0.3))`, which reaches `0.078` for the always-treat regime at the lowest
//! confounder and for the never-treat regime at the highest; heavy censoring censors each
//! person-period with probability `0.3` (about two thirds of the subjects are censored within
//! the three periods). Both keep the base hazard and confounder law, so the truth is the base
//! design's at the same horizon.
//!
//! These are design routes: they declare no graph and run no validation suite, so each record
//! is keyed on the design axes (structure `design`, validation `not_applicable`) and a
//! frequentist record without an interval binding in the runtime's key declares the interval
//! it publishes. The wide sequential studies measure coverage conditional on a returned interval
//! and count positivity refusals separately. Other studies count skips as misses.
//!
//! The designs are `#[ignore]`d measurement designs run by `scripts/measure_calibration.sh`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::{Arc, OnceLock};

use antecedent::{BayesianConfig, InferenceMode, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, SequentialContrastQuery, TemporalAssignment,
    TemporalDesignFunctional, TemporalDesignQuery, TemporalDoseTarget, VariableId,
};
use antecedent_data::TabularData;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, gaussian, grid_n,
    grid_seed, map_replicates, n_sim, stream_seed, uniform,
};
use common::calibration_bind::constructions;

/// Per-decision effect of the two-arm panel.
const TWO_ARM_EFFECT: f64 = 1.25;
/// Gaussian kernel bandwidth of every dose target.
const BANDWIDTH: f64 = 0.3;
/// Doses are uniform on `(0, DOSE_TOP)`.
const DOSE_TOP: f64 = 2.0;
/// Whole-procedure bootstrap replicates of the bootstrap routes: the route's replicate floor.
const BOOTSTRAP_REPLICATES: u32 = 99;
/// Horizon of the survival panels.
const SURVIVAL_HORIZON: u32 = 3;
/// Arm effects `0, 0.5, 1.5` of the multi-arm panel.
const MULTI_ARM_EFFECTS: [f64; 3] = [0.0, 0.5, 1.5];

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

fn flag(value: bool) -> f64 {
    f64::from(u8::from(value))
}

/// One route of the family that takes a decision lag.
#[derive(Clone, Copy, Debug)]
enum Kind {
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

/// A design: a route, its decision lag and whether its assignment probabilities sit on the
/// row's floor.
#[derive(Clone, Copy, Debug)]
struct Spec {
    kind: Kind,
    lag: usize,
    floor: bool,
}

impl Spec {
    /// Units (or interference clusters) at this run's sample-size grid point.
    fn units(self) -> usize {
        grid_n(match self.kind {
            Kind::MultiArm | Kind::Categorical => 250,
            Kind::Quantile => 200,
            _ => 300,
        })
    }

    /// Index and name of the scored coordinate.
    fn coordinate(self) -> (usize, &'static str) {
        match self.kind {
            Kind::TwoArm => (0, "excursion_effect"),
            Kind::MultiArm => (0, "contrast:mid-low"),
            Kind::Categorical => (2, "step:high-mid"),
            Kind::Factorial => (0, "main_effect:first"),
            Kind::Interference => (0, "direct_effect"),
            Kind::ContinuousDose => (0, "smoothed_contrast:1.2-0.8"),
            Kind::SmoothedDose => (1, "smoothed_response@1"),
            Kind::DoseGrid => (1, "level@1"),
            Kind::Quantile => (0, "quantile_effect"),
        }
    }

    /// The analytic value of the scored coordinate.
    fn truth(self) -> f64 {
        match self.kind {
            Kind::TwoArm => TWO_ARM_EFFECT,
            Kind::MultiArm => MULTI_ARM_EFFECTS[1] - MULTI_ARM_EFFECTS[0],
            Kind::Categorical => MULTI_ARM_EFFECTS[2] - MULTI_ARM_EFFECTS[1],
            // Averaged over the other factor's two levels with equal weight.
            Kind::Factorial => 1.0 + 0.5 * 0.4,
            Kind::Interference => 1.0,
            Kind::ContinuousDose => smoothed_response(1.2) - smoothed_response(0.8),
            Kind::SmoothedDose => smoothed_response(1.0),
            Kind::DoseGrid => dose_response(1.0),
            Kind::Quantile => 2.0,
        }
    }

    /// Whole-procedure bootstrap replicates a frequentist study of the route asks for.
    fn replicates(self) -> u32 {
        match self.kind {
            Kind::Quantile => BOOTSTRAP_REPLICATES,
            _ => 0,
        }
    }

    /// Posterior draws of a Bayesian study of the route.
    fn draws(self) -> usize {
        match self.kind {
            Kind::DoseGrid => 500,
            _ => 1000,
        }
    }

    /// The frequentist interval the route publishes: its method and standard-error kind.
    fn interval(self) -> (&'static str, &'static str) {
        match self.kind {
            Kind::DoseGrid => ("analytic_se", "local_quadratic_bias_aware"),
            Kind::Quantile => ("percentile_bootstrap", "percentile"),
            _ => ("analytic_se", "cluster_ratio"),
        }
    }

    /// Draw one panel and its query.
    fn draw(self, units: usize, seed: u64) -> (TabularData, TemporalDesignQuery) {
        let Self { lag, floor, .. } = self;
        match self.kind {
            Kind::TwoArm => two_arm(units, lag, floor, seed),
            Kind::MultiArm => multi_arm(units, lag, floor, false, seed),
            Kind::Categorical => multi_arm(units, lag, floor, true, seed),
            Kind::Factorial => factorial(units, lag, floor, seed),
            Kind::Interference => interference(units, lag, floor, seed),
            Kind::ContinuousDose => dose(
                units,
                lag,
                seed,
                TemporalDoseTarget::SmoothedContrast { active: 1.2, control: 0.8 },
            ),
            Kind::SmoothedDose => {
                dose(units, lag, seed, TemporalDoseTarget::SmoothedCurve { grid: dose_grid() })
            }
            Kind::DoseGrid => dose(
                units,
                lag,
                seed,
                TemporalDoseTarget::LocalQuadratic {
                    grid: dose_grid(),
                    derivative: false,
                    curvature_bound: Some(0.0),
                },
            ),
            Kind::Quantile => quantile(units, lag, floor, seed),
        }
    }
}

fn periods_of(base: i64, lag: usize) -> i64 {
    base + i64::try_from(lag).expect("a lag fits")
}

fn lag_u32(lag: usize) -> u32 {
    u32::try_from(lag).expect("a lag fits")
}

/// Decision `back` periods before period `t`, zero before the panel starts.
fn behind(history: &[f64], t: usize, back: usize) -> f64 {
    if t >= back { history[t - back] } else { 0.0 }
}

// ------------------------------------------------------------------ panels

/// The base two-arm panel with the effect landing `lag` periods after the decision (at least
/// one): the treatment probability depends on the previous outcome (`0.65` after a positive
/// outcome, else `0.35`; the floor design treats with probability `0.35` throughout), and the
/// outcome carries [`TWO_ARM_EFFECT`] of the decision `lag` periods before and `0.3` of the one
/// before that. The excursion effect of a decision on the outcome `lag` periods later is
/// [`TWO_ARM_EFFECT`].
fn two_arm(units: usize, lag: usize, floor: bool, seed: u64) -> (TabularData, TemporalDesignQuery) {
    let mut next = uniform(seed, 0x7D01);
    let (mut unit_labels, mut periods, mut assignment, mut probability, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let length = usize::try_from(periods_of(9, lag)).expect("a length fits");
    for i in 0..units {
        let level = next() - 0.5;
        let mut previous_y = 0.0_f64;
        let mut treated: Vec<f64> = Vec::new();
        for t in 0..length {
            let p = if floor || previous_y <= 0.0 { 0.35 } else { 0.65 };
            let a = next() < p;
            let y = level
                + TWO_ARM_EFFECT * behind(&treated, t, lag)
                + 0.3 * behind(&treated, t, lag + 1)
                + (next() - 0.5);
            unit_labels.push(format!("u{i}"));
            periods.push(i64::try_from(t).expect("a period fits"));
            assignment.push(a);
            probability.push(p);
            outcome.push(y);
            previous_y = y;
            treated.push(flag(a));
        }
    }
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&unit_labels),
        periods: periods.as_slice().into(),
        lag: lag_u32(lag),
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

/// The base three-arm panel with the arm effect landing `lag` periods after the decision. The
/// probability row alternates between `(0.5, 0.25, 0.25)` and `(0.2, 0.4, 0.4)`; the floor
/// design alternates `(0.6, 0.2, 0.2)` and `(0.2, 0.2, 0.6)`.
fn multi_arm(
    units: usize,
    lag: usize,
    floor: bool,
    categorical: bool,
    seed: u64,
) -> (TabularData, TemporalDesignQuery) {
    let mut next = uniform(seed, 0x7D02);
    let (mut unit_labels, mut periods, mut assignment, mut probabilities, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let length = usize::try_from(periods_of(6, lag)).expect("a length fits");
    for i in 0..units {
        let level = next() - 0.5;
        let mut arms: Vec<usize> = Vec::new();
        for t in 0..length {
            let row = match (floor, t % 2 == 0) {
                (false, true) => vec![0.5, 0.25, 0.25],
                (false, false) => vec![0.2, 0.4, 0.4],
                (true, true) => vec![0.6, 0.2, 0.2],
                (true, false) => vec![0.2, 0.2, 0.6],
            };
            let u = next();
            let arm = if u < row[0] {
                0_usize
            } else if u < row[0] + row[1] {
                1
            } else {
                2
            };
            // The arm `lag` periods before; a period before the panel's start has none, which
            // is the reference arm.
            let earlier = if lag == 0 {
                arm
            } else if t >= lag {
                arms[t - lag]
            } else {
                0
            };
            unit_labels.push(format!("u{i}"));
            periods.push(i64::try_from(t).expect("a period fits"));
            assignment.push(arm);
            probabilities.push(row);
            outcome.push(level + MULTI_ARM_EFFECTS[earlier] + (next() - 0.5));
            arms.push(arm);
        }
    }
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&unit_labels),
        periods: periods.as_slice().into(),
        lag: lag_u32(lag),
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

/// The base two-factor panel with the effects landing `lag` periods after the decisions. The
/// factors are assigned independently with probabilities `0.5` and `0.4` (the floor design
/// `0.4` and `0.5`, so the smallest joint cell probability is `0.2`); the outcome adds `1.0` for the first factor, `0.5` for the second and
/// `0.4` for both.
fn factorial(
    units: usize,
    lag: usize,
    floor: bool,
    seed: u64,
) -> (TabularData, TemporalDesignQuery) {
    let mut next = uniform(seed, 0x7D03);
    let (pa, pb) = if floor { (0.4, 0.5) } else { (0.5, 0.4) };
    let (mut unit_labels, mut periods, mut first, mut second, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let length = usize::try_from(periods_of(5, lag)).expect("a length fits");
    for i in 0..units {
        let (mut firsts, mut seconds): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        for t in 0..length {
            let a = next() < pa;
            let b = next() < pb;
            // The decisions `lag` periods before (the current ones at lag zero).
            let (fa, fb) = if lag == 0 {
                (flag(a), flag(b))
            } else {
                (behind(&firsts, t, lag), behind(&seconds, t, lag))
            };
            unit_labels.push(format!("u{i}"));
            periods.push(i64::try_from(t).expect("a period fits"));
            first.push(a);
            second.push(b);
            outcome.push(1.0 * fa + 0.5 * fb + 0.4 * fa * fb + (next() - 0.5));
            firsts.push(flag(a));
            seconds.push(flag(b));
        }
    }
    let rows = unit_labels.len();
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&unit_labels),
        periods: periods.as_slice().into(),
        lag: lag_u32(lag),
        assignment: TemporalAssignment::Factorial {
            first: first.as_slice().into(),
            first_probability: vec![pa; rows].into(),
            second: second.as_slice().into(),
            second_probability: vec![pb; rows].into(),
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.05,
        outcome_misclassification: None,
    };
    (table(&[("y", outcome.as_slice())]), query)
}

/// The base interference panel with the effects landing `lag` periods after the decisions:
/// pairs of units share a cluster; a unit's own treatment adds `1.0` and a treated peer adds
/// `0.6`, so the direct effect is `1.0`. Treatment probability `0.5` (the floor design `0.25`).
fn interference(
    clusters: usize,
    lag: usize,
    floor: bool,
    seed: u64,
) -> (TabularData, TemporalDesignQuery) {
    let mut next = uniform(seed, 0x7D04);
    let p = if floor { 0.25 } else { 0.5 };
    let length = usize::try_from(periods_of(4, lag)).expect("a length fits");
    let (mut unit_labels, mut periods, mut cluster_labels, mut assignment) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for c in 0..clusters {
        for t in 0..length {
            for member in 0..2 {
                unit_labels.push(format!("u{c}_{member}"));
                periods.push(i64::try_from(t).expect("a period fits"));
                cluster_labels.push(format!("c{c}"));
                assignment.push(next() < p);
            }
        }
    }
    let rows = unit_labels.len();
    let outcome: Vec<f64> = (0..rows)
        .map(|row| {
            // Rows run cluster, period, member, so the same member `lag` periods before is
            // `2 * lag` rows back and the peer is the other member of that pair.
            let t = (row / 2) % length;
            let earlier = if t >= lag { row - 2 * lag } else { usize::MAX };
            let signal = if earlier == usize::MAX {
                0.0
            } else {
                flag(assignment[earlier]) + 0.6 * flag(assignment[earlier ^ 1])
            };
            signal + (next() - 0.5)
        })
        .collect();
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&unit_labels),
        periods: periods.as_slice().into(),
        lag: lag_u32(lag),
        assignment: TemporalAssignment::Interference {
            assignment: assignment.as_slice().into(),
            probability: vec![p; rows].into(),
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
/// `m` of the dose `lag` periods before plus noise (`m(0)` before the panel's start).
fn dose(
    units: usize,
    lag: usize,
    seed: u64,
    target: TemporalDoseTarget,
) -> (TabularData, TemporalDesignQuery) {
    let mut next = uniform(seed, 0x7D05);
    let length = usize::try_from(periods_of(8, lag)).expect("a length fits");
    let (mut unit_labels, mut periods, mut doses, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..units {
        let mut earlier: Vec<f64> = Vec::new();
        for t in 0..length {
            let d = DOSE_TOP * next();
            let driving = if lag == 0 {
                d
            } else if t >= lag {
                earlier[t - lag]
            } else {
                0.0
            };
            unit_labels.push(format!("u{i}"));
            periods.push(i64::try_from(t).expect("a period fits"));
            doses.push(d);
            outcome.push(dose_response(driving) + (next() - 0.5));
            earlier.push(d);
        }
    }
    let rows = unit_labels.len();
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&unit_labels),
        periods: periods.as_slice().into(),
        lag: lag_u32(lag),
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

/// The base two-arm quantile panel with the shift landing `lag` periods after the decision:
/// the outcome is a uniform draw plus `2` when the decision `lag` periods before was treated,
/// so the median shift is exactly `2`. The treatment probability alternates between `0.4` and
/// `0.6` (the floor design treats with probability `0.4` throughout).
fn quantile(
    units: usize,
    lag: usize,
    floor: bool,
    seed: u64,
) -> (TabularData, TemporalDesignQuery) {
    let mut next = uniform(seed, 0x7D06);
    let length = usize::try_from(periods_of(6, lag)).expect("a length fits");
    let (mut unit_labels, mut periods, mut assignment, mut probability, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..units {
        let mut treated: Vec<f64> = Vec::new();
        for t in 0..length {
            let p = if floor || t % 2 == 0 { 0.4 } else { 0.6 };
            let a = next() < p;
            let current = flag(a);
            let driving = if lag == 0 { current } else { behind(&treated, t, lag) };
            unit_labels.push(format!("u{i}"));
            periods.push(i64::try_from(t).expect("a period fits"));
            assignment.push(a);
            probability.push(p);
            outcome.push(next() + 2.0 * driving);
            treated.push(current);
        }
    }
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&unit_labels),
        periods: periods.as_slice().into(),
        lag: lag_u32(lag),
        assignment: TemporalAssignment::TwoArm {
            assignment: assignment.as_slice().into(),
            probability: probability.as_slice().into(),
        },
        functional: TemporalDesignFunctional::Quantile { level: 0.5 },
        min_probability: 0.05,
        outcome_misclassification: None,
    };
    (table(&[("y", outcome.as_slice())]), query)
}

// --------------------------------------------------------------- execution

/// The constructions of a design-route execution and the interval it published, from a study
/// run on `data` and `query`; `None` when the study is refused or publishes no interval for
/// the coordinate.
struct Replicate {
    coordinate: String,
    interval: (f64, f64),
    construction: Construction,
    scope: ScopeFacts,
}

/// Run a temporal-design query as a frequentist or Bayesian study and read the interval at
/// `index`, keyed on the design axes; `declared` is the interval method and standard-error
/// kind a frequentist record declares beside the runtime's construction.
#[allow(clippy::too_many_arguments)]
fn execute_design(
    data: &TabularData,
    query: TemporalDesignQuery,
    replicates: u32,
    draws: usize,
    index: usize,
    declared: (&'static str, &'static str),
    bayesian: bool,
    seed: u64,
) -> Option<Replicate> {
    let ctx = ExecutionContext::for_tests(seed);
    let inference = if bayesian {
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(draws))
    } else {
        InferenceMode::Frequentist
    };
    let mut builder =
        Study::tabular(data.clone()).query(CausalQuery::TemporalDesign(query)).inference(inference);
    if !bayesian && replicates > 0 {
        builder = builder.bootstrap_replicates(replicates);
    }
    let study = builder.build().ok()?;
    let prepared = study.prepare(&ctx).ok()?;
    let result = prepared.estimate(data, &ctx).ok()?;
    let fit = result.temporal_design()?;
    let interval = (*fit.lower.get(index)?, *fit.upper.get(index)?);
    let contract = prepared.contract_for_result(&result).ok()?;
    let reported = constructions(&contract, &result);
    let (mut construction, mut scope, _) = if bayesian {
        reported.into_iter().find(|(c, _, _)| c.interval_method == "posterior_quantile")?
    } else {
        reported.into_iter().next()?
    };
    // A design route has no graph and runs no validation suite, so its record is keyed on
    // the design axes: structure `design`, no validation mode applies.
    construction.structure = "design".into();
    if !bayesian {
        construction.interval_method = declared.0.into();
        construction.se_kind = declared.1.into();
        construction.dependence = "cluster".into();
        construction.reported_level = REPORTED_LEVEL;
        scope.replicates_ok = fit.replicates.and_then(|(ok, _)| u32::try_from(ok).ok());
    }
    Some(Replicate {
        coordinate: fit.coordinates.get(index)?.to_string(),
        interval,
        construction,
        scope,
    })
}

/// Score the replicates of one record; a refusal or a missing interval is a skip, a miss.
fn score(
    test: &'static str,
    dgp: &'static str,
    interval: &'static str,
    truth: f64,
    expected: &str,
    runs: Vec<Option<Replicate>>,
) {
    let mut tally = CoverageTally::for_record(RecordKey { test, dgp, interval }, REPORTED_LEVEL);
    for run in runs {
        let Some(run) = run else {
            tally.skip();
            continue;
        };
        assert_eq!(run.coordinate, expected, "{test}: the scored coordinate");
        tally.bind_validation(&run.construction, run.scope, "not_applicable");
        tally.record(Some(run.interval), truth);
    }
    tally.assert();
}

/// Coverage of a lag or floor design's scored coordinate over `n_sim()` panels.
fn measure(test: &'static str, dgp: &'static str, spec: Spec, bayesian: bool, seed_base: u64) {
    let units = spec.units();
    let (index, name) = spec.coordinate();
    let interval = if bayesian { "posterior_quantile" } else { spec.interval().0 };
    let runs = map_replicates(n_sim(), |rep| {
        let seed = seed_base + rep;
        let (data, query) = spec.draw(units, seed);
        execute_design(
            &data,
            query,
            spec.replicates(),
            spec.draws(),
            index,
            spec.interval(),
            bayesian,
            seed,
        )
    });
    score(test, dgp, interval, spec.truth(), name, runs);
}

// ---------------------------------------------------- sequential contrast

/// Interval method of the interval a route forms from its own standard error.
const NORMAL_INTERVAL: &str = "analytic_se";
/// Interval method of the equal-tailed posterior interval.
const POSTERIOR_INTERVAL: &str = "posterior_quantile";
/// Posterior draws of the Bayesian sequential designs.
const SEQUENTIAL_DRAWS: usize = 2_000;
const SEQUENTIAL_BASE_N: usize = 400;
/// Contrast of the regime that treats at both decisions against the regime that treats at
/// neither: `E[Y(1, 1)] - E[Y(0, 0)] = (1 + 1.32 + 1.5) - 1`.
const CONTRAST_TRUTH: f64 = 1.32 + 1.5;

/// A draw on `[-sqrt 3, sqrt 3]` (mean zero, unit variance) from a uniform draw on `[0, 1)`.
fn bounded(unit: f64) -> f64 {
    3.0_f64.sqrt() * (2.0 * unit - 1.0)
}

/// Columns `l0, a0, l1, a1, y` of the base design with wider assignment models: `a0 ~
/// Bernoulli(expit(1.5 l0))` and `a1 ~ Bernoulli(expit(1.2 (l1 - .4)))`; the outcome model is
/// the base's, `y = 1 + .5 l0 + a0 + 1.5 a1 + .4 l1 + e` with `l1 = .5 l0 + .8 a0 + .5 u`.
fn wide_sequential_panel(n: usize, seed: u64) -> TabularData {
    let mut normal = gaussian(stream_seed(seed, 0x5E01));
    let mut unit = uniform(seed, 0x5E02);
    let mut history = uniform(seed, 0x5E03);
    let (mut l0s, mut a0s, mut l1s, mut a1s, mut ys) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let l0 = bounded(history());
        let a0 = flag(unit() < expit(1.5 * l0));
        let l1 = 0.5 * l0 + 0.8 * a0 + 0.5 * bounded(history());
        let a1 = flag(unit() < expit(1.2 * (l1 - 0.4)));
        let y = 1.0 + 0.5 * l0 + a0 + 1.5 * a1 + 0.4 * l1 + normal();
        l0s.push(l0);
        a0s.push(a0);
        l1s.push(l1);
        a1s.push(a1);
        ys.push(y);
    }
    table(&[
        ("l0", l0s.as_slice()),
        ("a0", a0s.as_slice()),
        ("l1", l1s.as_slice()),
        ("a1", a1s.as_slice()),
        ("y", ys.as_slice()),
    ])
}

fn sequential_query(n: usize) -> SequentialContrastQuery {
    SequentialContrastQuery {
        outcome: VariableId::from_raw(4),
        first_treatment: VariableId::from_raw(1),
        second_treatment: VariableId::from_raw(3),
        baseline: Arc::from([VariableId::from_raw(0)]),
        intermediate: Arc::from([VariableId::from_raw(2)]),
        active: [1, 1],
        control: [0, 0],
        fold_of_row: (0..n)
            .map(|row| u32::try_from(row % 5).expect("fits"))
            .collect::<Vec<_>>()
            .into(),
        fold_count: 5,
        min_probability: 0.05,
    }
}

/// One wide-probability sequential-contrast replicate; the frequentist interval is the
/// published Wald interval. A fitted probability outside `[0.05, 0.95]` is refused by the route
/// before an interval exists.
fn run_wide_sequential(seed: u64, bayesian: bool) -> Option<Replicate> {
    let n = grid_n(SEQUENTIAL_BASE_N);
    let data = wide_sequential_panel(n, seed);
    let builder =
        Study::tabular(data.clone()).query(CausalQuery::SequentialContrast(sequential_query(n)));
    let builder = if bayesian {
        builder.inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(SEQUENTIAL_DRAWS),
        ))
    } else {
        builder
    };
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = builder
        .build()
        .expect("valid sequential design")
        .prepare(&ctx)
        .expect("prepared sequential design");
    let result = match prepared.estimate(&data, &ctx) {
        Ok(result) => result,
        Err(error) if error.reason_code() == Some("positivity_violated") => return None,
        Err(error) => panic!("unexpected sequential estimate error: {error}"),
    };
    let contract = prepared.contract_for_result(&result).expect("reported sequential construction");
    let reported = constructions(&contract, &result);
    let (interval, (construction, scope, validation)) = if bayesian {
        let found = reported
            .into_iter()
            .find(|(construction, _, _)| {
                construction.interval_method == "posterior_absolute_deviation"
            })
            .expect("published symmetric posterior construction");
        let [lo, hi] = result
            .sequential_contrast()
            .expect("sequential contrast")
            .posterior_symmetric_interval
            .expect("published symmetric posterior interval");
        ((lo, hi), found)
    } else {
        let [lower, upper] = result
            .sequential_contrast()
            .expect("sequential result")
            .interval
            .expect("published Wald interval");
        ((lower, upper), reported.into_iter().next().expect("published Wald construction"))
    };
    assert_eq!(construction.query, "SequentialContrast");
    assert_eq!(construction.structure, "design");
    assert!(construction.functional.is_empty());
    assert_eq!(validation, "not_applicable");
    Some(Replicate { coordinate: String::new(), interval, construction, scope })
}

fn measure_wide_sequential(test: &'static str, bayesian: bool, seed: u64) {
    let interval = if bayesian { "posterior_absolute_deviation" } else { NORMAL_INTERVAL };
    // Each independently seeded replicate supplies exactly one returned interval.
    // A retry is allowed only for the typed positivity refusal in run_wide_sequential.
    let runs = map_replicates(n_sim(), |rep| {
        let initial = grid_seed(stream_seed(seed, rep));
        for refused in 0..1024 {
            let panel_seed = if refused == 0 { initial } else { stream_seed(initial, refused) };
            if let Some(run) = run_wide_sequential(panel_seed, bayesian) {
                return (refused, run);
            }
        }
        panic!("{test}: no interval after 1024 independent panels in replicate {rep}");
    });
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp: "sequential_panel_wide_probabilities", interval },
        REPORTED_LEVEL,
    );
    let mut refusals = CoverageTally::new(format!("{test}_overlap_refusals"), REPORTED_LEVEL);
    for (refused, run) in runs {
        let (lo, hi) = run.interval;
        assert!(lo.is_finite() && hi.is_finite() && lo <= hi, "{test}: invalid interval");
        tally.absorb(&refused);
        refusals.record(Some((0.0, f64::from(u32::try_from(refused).expect("retry limit")))), 0.0);
        tally.bind_validation(&run.construction, run.scope, "not_applicable");
        tally.record(Some(run.interval), CONTRAST_TRUTH);
    }
    tally.assert_with_overlap_refusals(&refusals);
}

// ----------------------------------------------------------- regime survival

/// A survival panel's law: the treatment probability is `expit(slope (L + shift))` and each
/// person-period is censored with probability `censoring`.
#[derive(Clone, Copy)]
struct SurvivalLaw {
    slope: f64,
    shift: f64,
    censoring: f64,
}

/// Low overlap: regime-consistent treatment probabilities down to about `0.078`.
const LOW_OVERLAP: SurvivalLaw = SurvivalLaw { slope: 1.9, shift: 0.3, censoring: 0.03 };
/// Heavy censoring: each person-period censored with probability `0.3`.
const HEAVY_CENSORING: SurvivalLaw = SurvivalLaw { slope: 0.8, shift: 0.0, censoring: 0.3 };

/// Confounder at period `k`: `L_k = 0.5 L_{k-1} - 0.4 A_{k-1} + (u - 0.5)`.
fn next_confounder(current: f64, previous_treated: bool, noise: f64) -> f64 {
    0.5 * current - 0.4 * flag(previous_treated) + (noise - 0.5)
}

/// Subject trajectories with time-varying confounding at the horizon of three periods: the base
/// design with the treatment and censoring laws of `law`; the hazard is
/// `expit(-2.5 + 0.8 L - 0.7 A)`, `L_0` is uniform on `(-1, 1)`.
fn survival(subjects: usize, law: SurvivalLaw, seed: u64) -> (TabularData, TemporalDesignQuery) {
    let mut next = uniform(seed, 0x7D07);
    let (mut unit_labels, mut periods) = (Vec::new(), Vec::new());
    let (mut events, mut treated, mut censored, mut confounder) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..subjects {
        let mut current = 2.0 * next() - 1.0;
        let mut previous = false;
        for k in 0..SURVIVAL_HORIZON {
            if k > 0 {
                current = next_confounder(current, previous, next());
            }
            let a = next() < expit(law.slope * (current + law.shift));
            let c = next() < law.censoring;
            let y = !c && next() < expit(-2.5 + 0.8 * current - 0.7 * flag(a));
            unit_labels.push(format!("s{i}"));
            periods.push(i64::from(k));
            events.push(flag(y));
            treated.push(flag(a));
            censored.push(flag(c));
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
            horizon: SURVIVAL_HORIZON,
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

/// Always-treat minus never-treat survival at the horizon, by simulating the generating law
/// under each static regime (the hazards are multiplied out, so only the confounder path is
/// random). Neither the treatment law nor the censoring law enters it.
fn survival_truth() -> f64 {
    static TRUTH: OnceLock<f64> = OnceLock::new();
    *TRUTH.get_or_init(|| {
        let draws = 2_000_000_u32;
        let mut next = uniform(0x5A17_0001, 0x7D08);
        let mut totals = [0.0_f64; 2];
        for (index, arm) in [true, false].into_iter().enumerate() {
            for _ in 0..draws {
                let mut current = 2.0 * next() - 1.0;
                let mut surviving = 1.0;
                for k in 0..SURVIVAL_HORIZON {
                    if k > 0 {
                        current = next_confounder(current, arm, next());
                    }
                    surviving *= 1.0 - expit(-2.5 + 0.8 * current - 0.7 * flag(arm));
                }
                totals[index] += surviving;
            }
        }
        (totals[0] - totals[1]) / f64::from(draws)
    })
}

fn measure_survival(
    test: &'static str,
    dgp: &'static str,
    law: SurvivalLaw,
    bayesian: bool,
    seed_base: u64,
) {
    let interval = if bayesian { POSTERIOR_INTERVAL } else { "percentile_bootstrap" };
    let truth = survival_truth();
    let runs = map_replicates(n_sim(), |rep| {
        let seed = seed_base + rep;
        let (data, query) = survival(grid_n(1200), law, seed);
        execute_design(
            &data,
            query,
            BOOTSTRAP_REPLICATES,
            500,
            0,
            ("percentile_bootstrap", "percentile"),
            bayesian,
            seed,
        )
    });
    score(test, dgp, interval, truth, "survival_difference", runs);
}

// ------------------------------------------------------------------- tests

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_two_arm_excursion_lag_two_interval() {
    measure(
        "temporal_design_two_arm_excursion_lag_two_interval",
        "sequential_two_arm_history_dependent_known_excursion_lag_two",
        Spec { kind: Kind::TwoArm, lag: 2, floor: false },
        false,
        0x7E00_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_two_arm_excursion_lag_two_interval() {
    measure(
        "bayesian_temporal_design_two_arm_excursion_lag_two_interval",
        "sequential_two_arm_history_dependent_known_excursion_lag_two",
        Spec { kind: Kind::TwoArm, lag: 2, floor: false },
        true,
        0x7E00_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_two_arm_excursion_lag_three_interval() {
    measure(
        "temporal_design_two_arm_excursion_lag_three_interval",
        "sequential_two_arm_history_dependent_known_excursion_lag_three",
        Spec { kind: Kind::TwoArm, lag: 3, floor: false },
        false,
        0x7E01_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_two_arm_excursion_lag_three_interval() {
    measure(
        "bayesian_temporal_design_two_arm_excursion_lag_three_interval",
        "sequential_two_arm_history_dependent_known_excursion_lag_three",
        Spec { kind: Kind::TwoArm, lag: 3, floor: false },
        true,
        0x7E01_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_multi_arm_contrast_lag_two_interval() {
    measure(
        "temporal_design_multi_arm_contrast_lag_two_interval",
        "sequential_three_arm_period_varying_probabilities_known_contrast_lag_two",
        Spec { kind: Kind::MultiArm, lag: 2, floor: false },
        false,
        0x7E02_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_multi_arm_contrast_lag_two_interval() {
    measure(
        "bayesian_temporal_design_multi_arm_contrast_lag_two_interval",
        "sequential_three_arm_period_varying_probabilities_known_contrast_lag_two",
        Spec { kind: Kind::MultiArm, lag: 2, floor: false },
        true,
        0x7E02_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_multi_arm_contrast_lag_three_interval() {
    measure(
        "temporal_design_multi_arm_contrast_lag_three_interval",
        "sequential_three_arm_period_varying_probabilities_known_contrast_lag_three",
        Spec { kind: Kind::MultiArm, lag: 3, floor: false },
        false,
        0x7E03_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_multi_arm_contrast_lag_three_interval() {
    measure(
        "bayesian_temporal_design_multi_arm_contrast_lag_three_interval",
        "sequential_three_arm_period_varying_probabilities_known_contrast_lag_three",
        Spec { kind: Kind::MultiArm, lag: 3, floor: false },
        true,
        0x7E03_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_categorical_step_lag_two_interval() {
    measure(
        "temporal_design_categorical_step_lag_two_interval",
        "sequential_three_level_period_varying_probabilities_known_ordered_step_lag_two",
        Spec { kind: Kind::Categorical, lag: 2, floor: false },
        false,
        0x7E04_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_categorical_step_lag_two_interval() {
    measure(
        "bayesian_temporal_design_categorical_step_lag_two_interval",
        "sequential_three_level_period_varying_probabilities_known_ordered_step_lag_two",
        Spec { kind: Kind::Categorical, lag: 2, floor: false },
        true,
        0x7E04_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_categorical_step_lag_three_interval() {
    measure(
        "temporal_design_categorical_step_lag_three_interval",
        "sequential_three_level_period_varying_probabilities_known_ordered_step_lag_three",
        Spec { kind: Kind::Categorical, lag: 3, floor: false },
        false,
        0x7E05_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_categorical_step_lag_three_interval() {
    measure(
        "bayesian_temporal_design_categorical_step_lag_three_interval",
        "sequential_three_level_period_varying_probabilities_known_ordered_step_lag_three",
        Spec { kind: Kind::Categorical, lag: 3, floor: false },
        true,
        0x7E05_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_factorial_main_effect_lag_two_interval() {
    measure(
        "temporal_design_factorial_main_effect_lag_two_interval",
        "sequential_two_factor_independent_assignment_known_main_effect_lag_two",
        Spec { kind: Kind::Factorial, lag: 2, floor: false },
        false,
        0x7E06_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_factorial_main_effect_lag_two_interval() {
    measure(
        "bayesian_temporal_design_factorial_main_effect_lag_two_interval",
        "sequential_two_factor_independent_assignment_known_main_effect_lag_two",
        Spec { kind: Kind::Factorial, lag: 2, floor: false },
        true,
        0x7E06_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_factorial_main_effect_lag_three_interval() {
    measure(
        "temporal_design_factorial_main_effect_lag_three_interval",
        "sequential_two_factor_independent_assignment_known_main_effect_lag_three",
        Spec { kind: Kind::Factorial, lag: 3, floor: false },
        false,
        0x7E07_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_factorial_main_effect_lag_three_interval() {
    measure(
        "bayesian_temporal_design_factorial_main_effect_lag_three_interval",
        "sequential_two_factor_independent_assignment_known_main_effect_lag_three",
        Spec { kind: Kind::Factorial, lag: 3, floor: false },
        true,
        0x7E07_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_interference_direct_effect_lag_two_interval() {
    measure(
        "temporal_design_interference_direct_effect_lag_two_interval",
        "sequential_paired_cluster_spillover_known_direct_effect_lag_two",
        Spec { kind: Kind::Interference, lag: 2, floor: false },
        false,
        0x7E08_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_interference_direct_effect_lag_two_interval() {
    measure(
        "bayesian_temporal_design_interference_direct_effect_lag_two_interval",
        "sequential_paired_cluster_spillover_known_direct_effect_lag_two",
        Spec { kind: Kind::Interference, lag: 2, floor: false },
        true,
        0x7E08_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_interference_direct_effect_lag_three_interval() {
    measure(
        "temporal_design_interference_direct_effect_lag_three_interval",
        "sequential_paired_cluster_spillover_known_direct_effect_lag_three",
        Spec { kind: Kind::Interference, lag: 3, floor: false },
        false,
        0x7E09_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_interference_direct_effect_lag_three_interval() {
    measure(
        "bayesian_temporal_design_interference_direct_effect_lag_three_interval",
        "sequential_paired_cluster_spillover_known_direct_effect_lag_three",
        Spec { kind: Kind::Interference, lag: 3, floor: false },
        true,
        0x7E09_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_continuous_dose_contrast_lag_two_interval() {
    measure(
        "temporal_design_continuous_dose_contrast_lag_two_interval",
        "uniform_dose_quadratic_response_known_smoothed_contrast_lag_two",
        Spec { kind: Kind::ContinuousDose, lag: 2, floor: false },
        false,
        0x7E0A_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_continuous_dose_contrast_lag_two_interval() {
    measure(
        "bayesian_temporal_design_continuous_dose_contrast_lag_two_interval",
        "uniform_dose_quadratic_response_known_smoothed_contrast_lag_two",
        Spec { kind: Kind::ContinuousDose, lag: 2, floor: false },
        true,
        0x7E0A_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_continuous_dose_contrast_lag_three_interval() {
    measure(
        "temporal_design_continuous_dose_contrast_lag_three_interval",
        "uniform_dose_quadratic_response_known_smoothed_contrast_lag_three",
        Spec { kind: Kind::ContinuousDose, lag: 3, floor: false },
        false,
        0x7E0B_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_continuous_dose_contrast_lag_three_interval() {
    measure(
        "bayesian_temporal_design_continuous_dose_contrast_lag_three_interval",
        "uniform_dose_quadratic_response_known_smoothed_contrast_lag_three",
        Spec { kind: Kind::ContinuousDose, lag: 3, floor: false },
        true,
        0x7E0B_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_smoothed_dose_curve_lag_two_interval() {
    measure(
        "temporal_design_smoothed_dose_curve_lag_two_interval",
        "uniform_dose_quadratic_response_known_smoothed_response_lag_two",
        Spec { kind: Kind::SmoothedDose, lag: 2, floor: false },
        false,
        0x7E0C_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_smoothed_dose_curve_lag_two_interval() {
    measure(
        "bayesian_temporal_design_smoothed_dose_curve_lag_two_interval",
        "uniform_dose_quadratic_response_known_smoothed_response_lag_two",
        Spec { kind: Kind::SmoothedDose, lag: 2, floor: false },
        true,
        0x7E0C_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_smoothed_dose_curve_lag_three_interval() {
    measure(
        "temporal_design_smoothed_dose_curve_lag_three_interval",
        "uniform_dose_quadratic_response_known_smoothed_response_lag_three",
        Spec { kind: Kind::SmoothedDose, lag: 3, floor: false },
        false,
        0x7E0D_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_smoothed_dose_curve_lag_three_interval() {
    measure(
        "bayesian_temporal_design_smoothed_dose_curve_lag_three_interval",
        "uniform_dose_quadratic_response_known_smoothed_response_lag_three",
        Spec { kind: Kind::SmoothedDose, lag: 3, floor: false },
        true,
        0x7E0D_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_dose_grid_local_quadratic_lag_two_interval() {
    measure(
        "temporal_design_dose_grid_local_quadratic_lag_two_interval",
        "uniform_dose_quadratic_response_known_level_zero_curvature_bound_lag_two",
        Spec { kind: Kind::DoseGrid, lag: 2, floor: false },
        false,
        0x7E0E_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_dose_grid_local_quadratic_lag_two_interval() {
    measure(
        "bayesian_temporal_design_dose_grid_local_quadratic_lag_two_interval",
        "uniform_dose_quadratic_response_known_level_zero_curvature_bound_lag_two",
        Spec { kind: Kind::DoseGrid, lag: 2, floor: false },
        true,
        0x7E0E_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_dose_grid_local_quadratic_lag_three_interval() {
    measure(
        "temporal_design_dose_grid_local_quadratic_lag_three_interval",
        "uniform_dose_quadratic_response_known_level_zero_curvature_bound_lag_three",
        Spec { kind: Kind::DoseGrid, lag: 3, floor: false },
        false,
        0x7E0F_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_dose_grid_local_quadratic_lag_three_interval() {
    measure(
        "bayesian_temporal_design_dose_grid_local_quadratic_lag_three_interval",
        "uniform_dose_quadratic_response_known_level_zero_curvature_bound_lag_three",
        Spec { kind: Kind::DoseGrid, lag: 3, floor: false },
        true,
        0x7E0F_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_quantile_effect_bootstrap_lag_two_interval() {
    measure(
        "temporal_design_quantile_effect_bootstrap_lag_two_interval",
        "sequential_two_arm_location_shift_known_median_shift_lag_two",
        Spec { kind: Kind::Quantile, lag: 2, floor: false },
        false,
        0x7E10_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_quantile_effect_lag_two_interval() {
    measure(
        "bayesian_temporal_design_quantile_effect_lag_two_interval",
        "sequential_two_arm_location_shift_known_median_shift_lag_two",
        Spec { kind: Kind::Quantile, lag: 2, floor: false },
        true,
        0x7E10_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_quantile_effect_bootstrap_lag_three_interval() {
    measure(
        "temporal_design_quantile_effect_bootstrap_lag_three_interval",
        "sequential_two_arm_location_shift_known_median_shift_lag_three",
        Spec { kind: Kind::Quantile, lag: 3, floor: false },
        false,
        0x7E11_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_quantile_effect_lag_three_interval() {
    measure(
        "bayesian_temporal_design_quantile_effect_lag_three_interval",
        "sequential_two_arm_location_shift_known_median_shift_lag_three",
        Spec { kind: Kind::Quantile, lag: 3, floor: false },
        true,
        0x7E11_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_two_arm_excursion_floor_probability_interval() {
    measure(
        "temporal_design_two_arm_excursion_floor_probability_interval",
        "sequential_two_arm_history_free_lag_one_probability_at_floor_known_excursion",
        Spec { kind: Kind::TwoArm, lag: 1, floor: true },
        false,
        0x7F00_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_two_arm_excursion_floor_probability_interval() {
    measure(
        "bayesian_temporal_design_two_arm_excursion_floor_probability_interval",
        "sequential_two_arm_history_free_lag_one_probability_at_floor_known_excursion",
        Spec { kind: Kind::TwoArm, lag: 1, floor: true },
        true,
        0x7F00_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_interference_direct_effect_floor_probability_interval() {
    measure(
        "temporal_design_interference_direct_effect_floor_probability_interval",
        "sequential_paired_cluster_spillover_probability_at_floor_known_direct_effect",
        Spec { kind: Kind::Interference, lag: 0, floor: true },
        false,
        0x7F01_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_interference_direct_effect_floor_probability_interval() {
    measure(
        "bayesian_temporal_design_interference_direct_effect_floor_probability_interval",
        "sequential_paired_cluster_spillover_probability_at_floor_known_direct_effect",
        Spec { kind: Kind::Interference, lag: 0, floor: true },
        true,
        0x7F01_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_multi_arm_contrast_floor_probability_interval() {
    measure(
        "temporal_design_multi_arm_contrast_floor_probability_interval",
        "sequential_three_arm_probabilities_at_floor_known_contrast",
        Spec { kind: Kind::MultiArm, lag: 0, floor: true },
        false,
        0x7F02_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_multi_arm_contrast_floor_probability_interval() {
    measure(
        "bayesian_temporal_design_multi_arm_contrast_floor_probability_interval",
        "sequential_three_arm_probabilities_at_floor_known_contrast",
        Spec { kind: Kind::MultiArm, lag: 0, floor: true },
        true,
        0x7F02_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_categorical_step_floor_probability_interval() {
    measure(
        "temporal_design_categorical_step_floor_probability_interval",
        "sequential_three_level_probabilities_at_floor_known_ordered_step",
        Spec { kind: Kind::Categorical, lag: 0, floor: true },
        false,
        0x7F03_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_categorical_step_floor_probability_interval() {
    measure(
        "bayesian_temporal_design_categorical_step_floor_probability_interval",
        "sequential_three_level_probabilities_at_floor_known_ordered_step",
        Spec { kind: Kind::Categorical, lag: 0, floor: true },
        true,
        0x7F03_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_factorial_main_effect_floor_probability_interval() {
    measure(
        "temporal_design_factorial_main_effect_floor_probability_interval",
        "sequential_two_factor_probabilities_at_floor_known_main_effect",
        Spec { kind: Kind::Factorial, lag: 0, floor: true },
        false,
        0x7F04_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_factorial_main_effect_floor_probability_interval() {
    measure(
        "bayesian_temporal_design_factorial_main_effect_floor_probability_interval",
        "sequential_two_factor_probabilities_at_floor_known_main_effect",
        Spec { kind: Kind::Factorial, lag: 0, floor: true },
        true,
        0x7F04_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_quantile_effect_bootstrap_floor_probability_interval() {
    measure(
        "temporal_design_quantile_effect_bootstrap_floor_probability_interval",
        "sequential_two_arm_probability_at_floor_known_median_shift",
        Spec { kind: Kind::Quantile, lag: 0, floor: true },
        false,
        0x7F05_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_quantile_effect_floor_probability_interval() {
    measure(
        "bayesian_temporal_design_quantile_effect_floor_probability_interval",
        "sequential_two_arm_probability_at_floor_known_median_shift",
        Spec { kind: Kind::Quantile, lag: 0, floor: true },
        true,
        0x7F05_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn sequential_crossfit_contrast_wald_interval_wide_probability_coverage() {
    measure_wide_sequential(
        "sequential_crossfit_contrast_wald_interval_wide_probability_coverage",
        false,
        0x5E2_0201,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_sequential_crossfit_contrast_posterior_wide_probability_coverage() {
    measure_wide_sequential(
        "bayesian_sequential_crossfit_contrast_posterior_wide_probability_coverage",
        true,
        0x5E2_0202,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_regime_survival_low_overlap_bootstrap_interval() {
    measure_survival(
        "temporal_design_regime_survival_low_overlap_bootstrap_interval",
        "time_varying_confounding_logistic_hazard_low_overlap_known_survival_difference",
        LOW_OVERLAP,
        false,
        0x7D31_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_regime_survival_heavy_censoring_bootstrap_interval() {
    measure_survival(
        "temporal_design_regime_survival_heavy_censoring_bootstrap_interval",
        "time_varying_confounding_logistic_hazard_heavy_censoring_known_survival_difference",
        HEAVY_CENSORING,
        false,
        0x7D32_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_regime_survival_low_overlap_interval() {
    measure_survival(
        "bayesian_temporal_design_regime_survival_low_overlap_interval",
        "time_varying_confounding_logistic_hazard_low_overlap_known_survival_difference",
        LOW_OVERLAP,
        true,
        0x7D33_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_regime_survival_heavy_censoring_interval() {
    measure_survival(
        "bayesian_temporal_design_regime_survival_heavy_censoring_interval",
        "time_varying_confounding_logistic_hazard_heavy_censoring_known_survival_difference",
        HEAVY_CENSORING,
        true,
        0x7D34_0000,
    );
}

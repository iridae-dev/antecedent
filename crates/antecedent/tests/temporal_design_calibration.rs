//! Coverage of the intervals of the sequentially randomized unit-period design routes and of
//! regime survival, each scored through the public study flow against a known truth.
//!
//! One measurement shape serves every route: draw a panel from a known law, run the route as a
//! frequentist study and as a Bayesian study, and score the 95% interval the route publishes
//! for one named coordinate against that coordinate's analytic value. The routes differ only
//! in the panel, the coordinate and the truth ([`Design`]):
//!
//! * two arms (lag one, history-dependent assignment, carryover inside the target): the
//!   excursion effect, truth the per-decision effect;
//! * several arms and categorical levels: the contrast of the middle arm against the reference
//!   and the ordered step from the middle to the top arm;
//! * the two-factor factorial: the main effect of the first factor, averaged over the other;
//! * within-cluster interference: the direct effect;
//! * the continuous dose with a uniform dose density: the kernel-smoothed contrast of two
//!   doses, the kernel-smoothed response, and the bias-aware local-quadratic level. The
//!   smoothed targets are scored against `int_0^2 K_h(u - d) m(u) du`, the quantity the
//!   inverse-density kernel score has as its expectation; the local-quadratic level is scored
//!   against `m(d)` with the third-derivative bound it declares (the curve is quadratic, so the
//!   bound is exact);
//! * the two-arm quantile excursion: the median shift of a location-shifted outcome;
//! * static-regime survival on subject trajectories with time-varying confounding: the
//!   always-treat minus never-treat survival difference at the horizon, truth by simulation of
//!   the generating law.
//!
//! The pointwise 95% interval of the first scored coordinate is the one scored. Bands and the
//! other coordinates are not claimed here. A replicate the route refuses is a skip (a miss).
//!
//! A frequentist result carries no interval binding in the runtime's calibration key (the
//! result is not an effect estimate), so its record declares the interval it publishes beside
//! the construction the runtime does report (query, estimator, identification, functional);
//! the Bayesian record reads the interval method from the runtime. Every record is keyed on
//! the design axes (structure `design`, validation `not_applicable`), as the randomized
//! design routes are: these routes have no graph and run no validation suite.
//!
//! Ignored tests run via `scripts/measure_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::{Arc, OnceLock};

use antecedent::{BayesianConfig, InferenceMode, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, TemporalAssignment, TemporalDesignFunctional,
    TemporalDesignQuery, TemporalDoseTarget, VariableId,
};
use antecedent_data::TabularData;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, map_replicates,
    n_sim, uniform,
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
/// Horizon of the survival panel.
const SURVIVAL_HORIZON: u32 = 3;

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
    RegimeSurvival,
}

impl Design {
    const ALL: [Self; 10] = [
        Self::TwoArm,
        Self::MultiArm,
        Self::Categorical,
        Self::Factorial,
        Self::Interference,
        Self::ContinuousDose,
        Self::SmoothedDose,
        Self::DoseGrid,
        Self::Quantile,
        Self::RegimeSurvival,
    ];

    /// Units (or interference clusters, or subjects) at this run's sample-size grid point.
    fn units(self) -> usize {
        grid_n(match self {
            Self::MultiArm | Self::Categorical => 250,
            Self::Quantile => 200,
            Self::RegimeSurvival => 1200,
            _ => 300,
        })
    }

    /// Index and name of the scored coordinate.
    fn coordinate(self) -> (usize, &'static str) {
        match self {
            Self::TwoArm => (0, "excursion_effect"),
            Self::MultiArm => (0, "contrast:mid-low"),
            Self::Categorical => (2, "step:high-mid"),
            Self::Factorial => (0, "main_effect:first"),
            Self::Interference => (0, "direct_effect"),
            Self::ContinuousDose => (0, "smoothed_contrast:1.2-0.8"),
            Self::SmoothedDose => (1, "smoothed_response@1"),
            Self::DoseGrid => (1, "level@1"),
            Self::Quantile => (0, "quantile_effect"),
            Self::RegimeSurvival => (0, "survival_difference"),
        }
    }

    /// The analytic value of the scored coordinate.
    fn truth(self) -> f64 {
        match self {
            Self::TwoArm => TWO_ARM_EFFECT,
            Self::MultiArm => MULTI_ARM_EFFECTS[1] - MULTI_ARM_EFFECTS[0],
            Self::Categorical => MULTI_ARM_EFFECTS[2] - MULTI_ARM_EFFECTS[1],
            // Averaged over the other factor's two levels with equal weight.
            Self::Factorial => 1.0 + 0.5 * 0.4,
            Self::Interference => 1.0,
            Self::ContinuousDose => smoothed_response(1.2) - smoothed_response(0.8),
            Self::SmoothedDose => smoothed_response(1.0),
            Self::DoseGrid => dose_response(1.0),
            Self::Quantile => 2.0,
            Self::RegimeSurvival => survival_truth(),
        }
    }

    /// Whole-procedure bootstrap replicates a frequentist study of the route asks for.
    fn replicates(self) -> u32 {
        match self {
            Self::Quantile | Self::RegimeSurvival => BOOTSTRAP_REPLICATES,
            _ => 0,
        }
    }

    /// Posterior draws of a Bayesian study of the route.
    fn draws(self) -> usize {
        match self {
            Self::DoseGrid | Self::RegimeSurvival => 500,
            _ => 1000,
        }
    }

    /// The frequentist interval the route publishes: its method and standard-error kind.
    fn interval(self) -> (&'static str, &'static str) {
        match self {
            Self::DoseGrid => ("analytic_se", "local_quadratic_bias_aware"),
            Self::Quantile | Self::RegimeSurvival => ("percentile_bootstrap", "percentile"),
            _ => ("analytic_se", "cluster_ratio"),
        }
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
            Self::RegimeSurvival => survival(units, seed),
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
            outcome.push(level + MULTI_ARM_EFFECTS[arm] + (next() - 0.5));
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
            outcome.push(1.0 * fa + 0.5 * fb + 0.4 * fa * fb + (next() - 0.5));
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
            f64::from(u8::from(assignment[row]))
                + 0.6 * f64::from(u8::from(assignment[peer]))
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
    let mut next = uniform(seed, 0x7D05);
    let (mut unit_labels, mut periods, mut doses, mut outcome) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..units {
        for t in 0..8_i64 {
            let d = DOSE_TOP * next();
            unit_labels.push(format!("u{i}"));
            periods.push(t);
            doses.push(d);
            outcome.push(dose_response(d) + (next() - 0.5));
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
        functional: TemporalDesignFunctional::Quantile { level: 0.5 },
        min_probability: 0.05,
        outcome_misclassification: None,
    };
    (table(&[("y", outcome.as_slice())]), query)
}

/// Confounder at period `k`: `L_k = 0.5 L_{k-1} - 0.4 A_{k-1} + (u - 0.5)`.
fn next_confounder(current: f64, previous_treated: bool, noise: f64) -> f64 {
    0.5 * current - 0.4 * f64::from(u8::from(previous_treated)) + (noise - 0.5)
}

/// Subject trajectories with time-varying confounding: treatment is more likely when the
/// confounder is high, the hazard is `expit(-2.5 + 0.8 L - 0.7 A)` and censoring is light.
fn survival(subjects: usize, seed: u64) -> (TabularData, TemporalDesignQuery) {
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
/// random).
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
                    surviving *= 1.0 - expit(-2.5 + 0.8 * current - 0.7 * f64::from(u8::from(arm)));
                }
                totals[index] += surviving;
            }
        }
        (totals[0] - totals[1]) / f64::from(draws)
    })
}

// --------------------------------------------------------------- execution

/// One replicate: the scored coordinate, its published interval and the construction the
/// execution is keyed under.
struct Replicate {
    coordinate: String,
    estimate: f64,
    standard_error: f64,
    interval: (f64, f64),
    construction: Construction,
    scope: ScopeFacts,
    validation: &'static str,
}

/// Run the route on one simulated panel; `None` when the study is refused or publishes no
/// interval for the scored coordinate.
fn execute(design: Design, bayesian: bool, units: usize, seed: u64) -> Option<Replicate> {
    let (data, query) = design.draw(units, seed);
    let ctx = ExecutionContext::for_tests(seed);
    let inference = if bayesian {
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(design.draws()))
    } else {
        InferenceMode::Frequentist
    };
    let mut builder =
        Study::tabular(data.clone()).query(CausalQuery::TemporalDesign(query)).inference(inference);
    if !bayesian && design.replicates() > 0 {
        builder = builder.bootstrap_replicates(design.replicates());
    }
    let study = builder.build().ok()?;
    let prepared = study.prepare(&ctx).ok()?;
    let result = prepared.estimate(&data, &ctx).ok()?;
    let fit = result.temporal_design()?;
    let (index, _) = design.coordinate();
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
        let (method, se_kind) = design.interval();
        construction.interval_method = method.into();
        construction.se_kind = se_kind.into();
        construction.dependence = "cluster".into();
        construction.reported_level = REPORTED_LEVEL;
        scope.replicates_ok = fit.replicates.and_then(|(ok, _)| u32::try_from(ok).ok());
    }
    Some(Replicate {
        coordinate: fit.coordinates.get(index)?.to_string(),
        estimate: *fit.estimate.get(index)?,
        standard_error: *fit.standard_error.get(index)?,
        interval,
        construction,
        scope,
        validation: "not_applicable",
    })
}

/// Coverage of the scored coordinate's 95% interval over `n_sim()` panels.
fn measure(test: &'static str, dgp: &'static str, design: Design, bayesian: bool, seed_base: u64) {
    let units = design.units();
    let interval = if bayesian { "posterior_quantile" } else { design.interval().0 };
    let mut tally = CoverageTally::for_record(RecordKey { test, dgp, interval }, REPORTED_LEVEL);
    let truth = design.truth();
    let runs = map_replicates(n_sim(), |rep| execute(design, bayesian, units, seed_base + rep));
    for run in runs {
        // A refusal or a missing interval is a miss, never a free pass.
        let Some(run) = run else {
            tally.skip();
            continue;
        };
        tally.bind_validation(&run.construction, run.scope, run.validation);
        tally.record(Some(run.interval), truth);
    }
    tally.assert();
}

/// One panel per route, quickly: the scored coordinate is the one named here, its estimate sits
/// near the known truth and the published interval is ordered and finite.
#[test]
fn every_route_scores_its_named_coordinate_near_the_known_truth() {
    for design in Design::ALL {
        let run = execute(design, false, design.units(), 0x7D00_0001)
            .unwrap_or_else(|| panic!("{design:?}: the study runs and publishes an interval"));
        assert_eq!(run.coordinate, design.coordinate().1, "{design:?}");
        let (lower, upper) = run.interval;
        assert!(lower.is_finite() && upper.is_finite() && lower <= upper, "{design:?}");
        assert!(run.standard_error.is_finite() && run.standard_error > 0.0, "{design:?}");
        let truth = design.truth();
        assert!(
            (run.estimate - truth).abs() < 5.0 * run.standard_error + 0.01,
            "{design:?}: estimate {} vs truth {truth}, se {}",
            run.estimate,
            run.standard_error
        );
    }
}

// ------------------------------------------------------------- frequentist

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_two_arm_excursion_interval() {
    measure(
        "temporal_design_two_arm_excursion_interval",
        "sequential_two_arm_history_dependent_lag_one_known_excursion",
        Design::TwoArm,
        false,
        0x7D10_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_multi_arm_contrast_interval() {
    measure(
        "temporal_design_multi_arm_contrast_interval",
        "sequential_three_arm_period_varying_probabilities_known_contrast",
        Design::MultiArm,
        false,
        0x7D11_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_categorical_step_interval() {
    measure(
        "temporal_design_categorical_step_interval",
        "sequential_three_level_period_varying_probabilities_known_ordered_step",
        Design::Categorical,
        false,
        0x7D12_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_factorial_main_effect_interval() {
    measure(
        "temporal_design_factorial_main_effect_interval",
        "sequential_two_factor_independent_assignment_known_main_effect",
        Design::Factorial,
        false,
        0x7D13_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_interference_direct_effect_interval() {
    measure(
        "temporal_design_interference_direct_effect_interval",
        "sequential_paired_cluster_spillover_known_direct_effect",
        Design::Interference,
        false,
        0x7D14_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_continuous_dose_contrast_interval() {
    measure(
        "temporal_design_continuous_dose_contrast_interval",
        "uniform_dose_quadratic_response_known_smoothed_contrast",
        Design::ContinuousDose,
        false,
        0x7D15_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_smoothed_dose_curve_interval() {
    measure(
        "temporal_design_smoothed_dose_curve_interval",
        "uniform_dose_quadratic_response_known_smoothed_response",
        Design::SmoothedDose,
        false,
        0x7D16_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_dose_grid_local_quadratic_interval() {
    measure(
        "temporal_design_dose_grid_local_quadratic_interval",
        "uniform_dose_quadratic_response_known_level_zero_curvature_bound",
        Design::DoseGrid,
        false,
        0x7D17_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_quantile_effect_bootstrap_interval() {
    measure(
        "temporal_design_quantile_effect_bootstrap_interval",
        "sequential_two_arm_location_shift_known_median_shift",
        Design::Quantile,
        false,
        0x7D18_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_design_regime_survival_bootstrap_interval() {
    measure(
        "temporal_design_regime_survival_bootstrap_interval",
        "time_varying_confounding_logistic_hazard_known_survival_difference",
        Design::RegimeSurvival,
        false,
        0x7D19_0000,
    );
}

// ---------------------------------------------------------------- Bayesian

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_two_arm_excursion_interval() {
    measure(
        "bayesian_temporal_design_two_arm_excursion_interval",
        "sequential_two_arm_history_dependent_lag_one_known_excursion",
        Design::TwoArm,
        true,
        0x7D10_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_multi_arm_contrast_interval() {
    measure(
        "bayesian_temporal_design_multi_arm_contrast_interval",
        "sequential_three_arm_period_varying_probabilities_known_contrast",
        Design::MultiArm,
        true,
        0x7D11_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_categorical_step_interval() {
    measure(
        "bayesian_temporal_design_categorical_step_interval",
        "sequential_three_level_period_varying_probabilities_known_ordered_step",
        Design::Categorical,
        true,
        0x7D12_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_factorial_main_effect_interval() {
    measure(
        "bayesian_temporal_design_factorial_main_effect_interval",
        "sequential_two_factor_independent_assignment_known_main_effect",
        Design::Factorial,
        true,
        0x7D13_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_interference_direct_effect_interval() {
    measure(
        "bayesian_temporal_design_interference_direct_effect_interval",
        "sequential_paired_cluster_spillover_known_direct_effect",
        Design::Interference,
        true,
        0x7D14_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_continuous_dose_contrast_interval() {
    measure(
        "bayesian_temporal_design_continuous_dose_contrast_interval",
        "uniform_dose_quadratic_response_known_smoothed_contrast",
        Design::ContinuousDose,
        true,
        0x7D15_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_smoothed_dose_curve_interval() {
    measure(
        "bayesian_temporal_design_smoothed_dose_curve_interval",
        "uniform_dose_quadratic_response_known_smoothed_response",
        Design::SmoothedDose,
        true,
        0x7D16_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_dose_grid_local_quadratic_interval() {
    measure(
        "bayesian_temporal_design_dose_grid_local_quadratic_interval",
        "uniform_dose_quadratic_response_known_level_zero_curvature_bound",
        Design::DoseGrid,
        true,
        0x7D17_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_quantile_effect_interval() {
    measure(
        "bayesian_temporal_design_quantile_effect_interval",
        "sequential_two_arm_location_shift_known_median_shift",
        Design::Quantile,
        true,
        0x7D18_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_design_regime_survival_interval() {
    measure(
        "bayesian_temporal_design_regime_survival_interval",
        "time_varying_confounding_logistic_hazard_known_survival_difference",
        Design::RegimeSurvival,
        true,
        0x7D19_0000,
    );
}

//! Planted-truth data for the measured-exposure route: an exposure caused by a covariate, an
//! outcome with a known exposure effect, and the exposure recorded with error.
//!
//! Columns are `z, x, y` (covariate, recorded or latent-facet exposure, outcome) so the
//! confounded graph is `z -> x`, `z -> y`, `x -> y` on variable ids 0, 1, 2. Each scenario
//! returns the table, the declaration and the planted effect.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    dead_code,
    reason = "bounded simulation sizes and small indices; 0/1 indicators compared exactly; shared by several test binaries, each of which uses a subset"
)]

use antecedent::estimate::measurement::{
    DetectionFunctionDeclaration, ExposureErrorDeclaration, ExposureMeasurement, RatePair,
};
use antecedent::measured_exposure::{MeasuredColumns, MeasuredExposureRequest};
use antecedent::{CausalError, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{AverageEffectQuery, CausalQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

use super::calibration::{gaussian, uniform};

/// Which declaration a scenario plants.
#[derive(Clone, Copy, Debug)]
pub enum Scenario {
    /// One sensitivity and specificity, a continuous outcome.
    Shared,
    /// Outcome-specific rates, a 0/1 outcome.
    Differential,
    /// Outcome-specific rates, a 0/1 outcome, declared the way a validation study reports them:
    /// each declared rate is the planted rate plus an estimation error of the declared variance,
    /// so the declared variance is the true uncertainty of the declared rate.
    DifferentialValidated,
    /// Calibrated per-unit probabilities, a continuous outcome.
    Soft,
    /// Covariate-dependent detection, a continuous outcome.
    Detected,
    /// Three repeat codings of a latent binary facet, a continuous outcome.
    Repeat,
}

/// A planted dataset.
pub struct Planted {
    /// The analysis table's columns.
    pub columns: MeasuredColumns,
    /// The declaration on the exposure column `x`.
    pub declaration: ExposureMeasurement,
    /// The planted effect of the exposure.
    pub truth: f64,
    /// Variable id of the exposure in the completed tables.
    pub treatment: u32,
    /// Variable id of the outcome.
    pub outcome: u32,
    /// Directed edges of the adjustment graph over the table's variable ids.
    pub edges: Vec<(u32, u32)>,
}

/// Column order `z, x, y`: `z -> x`, `z -> y`, `x -> y`.
fn recorded_ids() -> (u32, u32, Vec<(u32, u32)>) {
    (1, 2, vec![(0, 1), (0, 2), (1, 2)])
}

/// Column order `z, y, x` (the exposure column is appended to the completed tables).
fn appended_ids() -> (u32, u32, Vec<(u32, u32)>) {
    (2, 1, vec![(0, 2), (0, 1), (2, 1)])
}

/// The ordinary frequentist average effect of the planted exposure on one completed table.
///
/// # Errors
/// The study's own refusal.
pub fn analyse(
    planted: &Planted,
    table: &TabularData,
    ctx: &ExecutionContext,
) -> Result<StudyResult, CausalError> {
    let mut graph = Dag::with_variables(3 + u32::from(planted.columns.has_column("count")));
    for &(from, to) in &planted.edges {
        graph
            .insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
            .expect("a planted edge is acyclic");
    }
    Study::tabular(table.clone())
        .graph(graph)
        .query(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(planted.treatment),
            VariableId::from_raw(planted.outcome),
        )))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(ctx)
}

fn expit(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

/// The planted risk difference of `expit(-0.5 + x + 0.4 z)`, integrated over `z ~ N(0, 1)`.
fn differential_truth() -> f64 {
    let steps = 4000;
    let (lo, hi) = (-8.0_f64, 8.0_f64);
    let h = (hi - lo) / f64::from(steps);
    (0..=steps)
        .map(|k| {
            let z = lo + f64::from(k) * h;
            let weight = if k == 0 || k == steps { 0.5 } else { 1.0 };
            let density = (-0.5 * z * z).exp() / (std::f64::consts::TAU).sqrt();
            weight * density * (expit(-0.5 + 1.0 + 0.4 * z) - expit(-0.5 + 0.4 * z)) * h
        })
        .sum()
}

fn poisson(next: &mut impl FnMut() -> f64, mean: f64) -> f64 {
    let limit = (-mean).exp();
    let (mut count, mut product) = (0.0, next());
    while product > limit {
        count += 1.0;
        product *= next();
    }
    count
}

/// Plant `n` units of `scenario` from `seed`.
///
/// # Panics
/// Never for the fixed rates and shapes used here.
#[must_use]
#[allow(clippy::too_many_lines, reason = "one planted scenario per declaration")]
pub fn plant(scenario: Scenario, n: usize, seed: u64) -> Planted {
    let mut normal = gaussian(seed);
    let mut next = uniform(seed, 1);
    let z: Vec<f64> = (0..n).map(|_| normal()).collect();
    let x: Vec<f64> = z.iter().map(|&zi| f64::from(u8::from(next() < expit(0.3 * zi)))).collect();
    let names = |extra: &[&str]| -> Vec<String> {
        let mut v: Vec<String> = ["z", "x", "y"].iter().map(|s| (*s).to_owned()).collect();
        v.extend(extra.iter().map(|s| (*s).to_owned()));
        v
    };
    match scenario {
        Scenario::Shared => {
            let (se, sp) = (0.85, 0.9);
            let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * x[i] + 0.5 * z[i] + normal()).collect();
            let recorded: Vec<bool> =
                (0..n).map(|i| if x[i] == 1.0 { next() < se } else { next() >= sp }).collect();
            let w: Vec<f64> = recorded.iter().map(|&b| f64::from(u8::from(b))).collect();
            Planted {
                columns: MeasuredColumns::new(names(&[]), vec![z, w, y]).unwrap(),
                declaration: ExposureMeasurement::Misclassified {
                    recorded,
                    error: ExposureErrorDeclaration::Shared(
                        RatePair::declared(se, sp, 1e-4, 1e-4).unwrap(),
                    ),
                    min_youden: 0.1,
                },
                truth: 2.0,
                treatment: 1,
                outcome: 2,
                edges: recorded_ids().2,
            }
        }
        Scenario::Differential | Scenario::DifferentialValidated => {
            let y: Vec<f64> = (0..n)
                .map(|i| f64::from(u8::from(next() < expit(-0.5 + x[i] + 0.4 * z[i]))))
                .collect();
            let (cases, controls) = ((0.9, 0.8), (0.7, 0.95));
            let recorded: Vec<bool> = (0..n)
                .map(|i| {
                    let (se, sp) = if y[i] == 1.0 { cases } else { controls };
                    if x[i] == 1.0 { next() < se } else { next() >= sp }
                })
                .collect();
            let w: Vec<f64> = recorded.iter().map(|&b| f64::from(u8::from(b))).collect();
            // The data above use the planted rates. A validation study reports estimates of
            // them with an error of the declared variance; declaring the planted rates exactly
            // while carrying that variance adds uncertainty the data never had, which does not
            // shrink with `n`.
            let mut declared = |rate: f64| {
                if matches!(scenario, Scenario::DifferentialValidated) {
                    (rate + 1e-4_f64.sqrt() * normal()).clamp(0.05, 0.995)
                } else {
                    rate
                }
            };
            let (case_se, case_sp) = (declared(cases.0), declared(cases.1));
            let (control_se, control_sp) = (declared(controls.0), declared(controls.1));
            Planted {
                columns: MeasuredColumns::new(names(&[]), vec![z, w, y]).unwrap(),
                declaration: ExposureMeasurement::Misclassified {
                    recorded,
                    error: ExposureErrorDeclaration::ByOutcome {
                        cases: RatePair::declared(case_se, case_sp, 1e-4, 1e-4).unwrap(),
                        controls: RatePair::declared(control_se, control_sp, 1e-4, 1e-4).unwrap(),
                    },
                    min_youden: 0.1,
                },
                truth: differential_truth(),
                treatment: 1,
                outcome: 2,
                edges: recorded_ids().2,
            }
        }
        Scenario::Soft => {
            let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * x[i] + 0.5 * z[i] + normal()).collect();
            // A signal N(+-1.5, 1) around the exposure; the probability is exact given z and it.
            let probability: Vec<f64> = (0..n)
                .map(|i| {
                    let signal = normal() + 1.5 * (2.0 * x[i] - 1.0);
                    let prior = expit(0.3 * z[i]);
                    let odds = prior / (1.0 - prior) * (3.0 * signal).exp();
                    odds / (1.0 + odds)
                })
                .collect();
            // The exposure is not recorded: its column is the draws, so the table has none.
            Planted {
                columns: MeasuredColumns::new(vec!["z".into(), "y".into()], vec![z, y]).unwrap(),
                declaration: ExposureMeasurement::Soft { probability },
                truth: 2.0,
                treatment: appended_ids().0,
                outcome: appended_ids().1,
                edges: appended_ids().2,
            }
        }
        Scenario::Detected => {
            let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * x[i] + 0.5 * z[i] + normal()).collect();
            let count: Vec<f64> = (0..n).map(|_| poisson(&mut next, 3.0)).collect();
            let detected: Vec<bool> =
                (0..n).map(|i| x[i] == 1.0 && next() < expit(-1.0 + 0.8 * count[i])).collect();
            let w: Vec<f64> = detected.iter().map(|&b| f64::from(u8::from(b))).collect();
            Planted {
                columns: MeasuredColumns::new(names(&["count"]), vec![z, w, y, count.clone()])
                    .unwrap(),
                declaration: ExposureMeasurement::Detected {
                    detected,
                    covariate: count,
                    function: DetectionFunctionDeclaration::Logistic {
                        intercept: -1.0,
                        slope: 0.8,
                        intercept_variance: 1e-3,
                        covariance: 0.0,
                        slope_variance: 1e-3,
                    },
                    false_positive_rate: 0.0,
                    false_positive_variance: 0.0,
                    min_net_detection: 0.1,
                },
                truth: 2.0,
                treatment: 1,
                outcome: 2,
                edges: recorded_ids().2,
            }
        }
        Scenario::Repeat => {
            let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * x[i] + 0.5 * z[i] + normal()).collect();
            let mut codings = Vec::with_capacity(3 * n);
            for i in 0..n {
                for _ in 0..3 {
                    let p = if x[i] == 1.0 { 0.8 } else { 0.15 };
                    codings.push(u8::from(next() < p));
                }
            }
            Planted {
                columns: MeasuredColumns::new(vec!["z".into(), "y".into()], vec![z, y]).unwrap(),
                declaration: ExposureMeasurement::RepeatCoded {
                    codings,
                    coding_count: 3,
                    categories: 2,
                    latent_levels: 2,
                    exchangeable: false,
                    lowest_level: None,
                    starts: 8,
                },
                truth: 2.0,
                treatment: appended_ids().0,
                outcome: appended_ids().1,
                edges: appended_ids().2,
            }
        }
    }
}

/// The request for a planted scenario: the exposure column is `x`, the outcome `y`, the
/// imputation conditions on `z`.
#[must_use]
pub fn request(draws: usize, seed: u64) -> MeasuredExposureRequest {
    MeasuredExposureRequest {
        exposure: "x".into(),
        outcome: "y".into(),
        covariates: vec!["z".into()],
        draws,
        seed,
    }
}

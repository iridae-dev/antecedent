//! Generating processes with closed-form truths for the observational survival and policy
//! value routes, shared by their known-truth tests and calibration designs.
//!
//! Survival: exponential event times `T(a)` with rate `0.25 exp(-0.5 a + 0.7 z1 + 0.5 z2)`,
//! two discrete confounders (`z1` Bernoulli(1/2), `z2` uniform on `{-1, 0, 1}`) that also
//! drive treatment, and exponential censoring. The survival probability and restricted mean
//! survival time of each arm are finite sums over the six confounder cells.
//!
//! Policy: one standard normal confounder `z` with `E[Y(0) | z] = 1 + z`, effect
//! `slope * z`, treatment `expit(steepness * z)` and independent normal noise; the value of
//! "treat when `z > c`" is `1 + slope * phi(c)` with `phi` the standard normal density.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(dead_code, reason = "shared by several test binaries, each of which uses a subset")]

use antecedent_core::CausalRng;
use antecedent_kernels::{sigmoid, standard_normal};

pub const HORIZON: f64 = 2.0;
const BASELINE_RATE: f64 = 0.25;
const TREATMENT_LOG_HAZARD: f64 = -0.5;
const Z1_LOG_HAZARD: f64 = 0.7;
const Z2_LOG_HAZARD: f64 = 0.5;

/// Standard normal density at zero.
pub const PHI_0: f64 = 0.398_942_280_401_432_7;

/// Standard normal density.
pub fn phi(x: f64) -> f64 {
    PHI_0 * (-0.5 * x * x).exp()
}

#[derive(Clone, Copy)]
pub enum Censoring {
    /// A constant censoring rate for everyone.
    Independent(f64),
    /// A censoring rate that depends on the arm and both confounders.
    DependsOnCovariates(f64),
}

pub struct SurvivalStudy {
    pub duration: Vec<f64>,
    pub event: Vec<f64>,
    pub treatment: Vec<f64>,
    pub z1: Vec<f64>,
    pub z2: Vec<f64>,
}

pub fn event_rate(treated: f64, z1: f64, z2: f64) -> f64 {
    BASELINE_RATE * (TREATMENT_LOG_HAZARD * treated + Z1_LOG_HAZARD * z1 + Z2_LOG_HAZARD * z2).exp()
}

pub fn survival_study(
    n: usize,
    seed: u64,
    censoring: Censoring,
    propensity_slope: f64,
) -> SurvivalStudy {
    let mut rng = CausalRng::from_seed(seed);
    let mut out = SurvivalStudy {
        duration: Vec::with_capacity(n),
        event: Vec::with_capacity(n),
        treatment: Vec::with_capacity(n),
        z1: Vec::with_capacity(n),
        z2: Vec::with_capacity(n),
    };
    for _ in 0..n {
        let z1 = f64::from(u8::from(rng.next_f64() < 0.5));
        let z2 = (rng.next_f64() * 3.0).floor() - 1.0;
        let treated = f64::from(u8::from(
            rng.next_f64() < sigmoid(propensity_slope * (-0.2 + 0.9 * z1 + 0.6 * z2)),
        ));
        let event_time = -rng.next_f64().max(1e-300).ln() / event_rate(treated, z1, z2);
        let censor_rate = match censoring {
            Censoring::Independent(rate) => rate,
            Censoring::DependsOnCovariates(rate) => {
                rate * (0.9 * z1 - 0.6 * z2 + 0.4 * treated).exp()
            }
        };
        let censor_time = -rng.next_f64().max(1e-300).ln() / censor_rate;
        out.duration.push(event_time.min(censor_time));
        out.event.push(f64::from(u8::from(event_time <= censor_time)));
        out.treatment.push(treated);
        out.z1.push(z1);
        out.z2.push(z2);
    }
    out
}

/// `(S_a(t), R_a(t))` averaged over the six equally likely confounder cells.
pub fn survival_truth(treated: f64, horizon: f64) -> (f64, f64) {
    let (mut survival, mut rmst) = (0.0, 0.0);
    for z1 in [0.0, 1.0] {
        for z2 in [-1.0, 0.0, 1.0] {
            let rate = event_rate(treated, z1, z2);
            survival += (-rate * horizon).exp() / 6.0;
            rmst += (1.0 - (-rate * horizon).exp()) / rate / 6.0;
        }
    }
    (survival, rmst)
}

pub struct BinaryStudy {
    pub outcome: Vec<f64>,
    pub treatment: Vec<f64>,
    pub z: Vec<f64>,
    pub noise: Vec<Vec<f64>>,
}

/// `E[Y(0) | z] = 1 + z`, effect `slope * z`, treatment `expit(steepness * z)`;
/// `noise_columns` covariates are unrelated to everything.
pub fn binary_study(
    n: usize,
    seed: u64,
    slope: f64,
    noise_columns: usize,
    steepness: f64,
) -> BinaryStudy {
    offset_binary_study(n, seed, slope, noise_columns, steepness, 0.0)
}

/// [`binary_study`] with treatment probability `expit(steepness z + offset)`, which makes the
/// assignment asymmetric in `z`.
pub fn offset_binary_study(
    n: usize,
    seed: u64,
    slope: f64,
    noise_columns: usize,
    steepness: f64,
    offset: f64,
) -> BinaryStudy {
    let mut rng = CausalRng::from_seed(seed);
    let mut out = BinaryStudy {
        outcome: Vec::new(),
        treatment: Vec::new(),
        z: Vec::new(),
        noise: vec![Vec::new(); noise_columns],
    };
    for _ in 0..n {
        let z = standard_normal(&mut rng);
        let treated = f64::from(u8::from(rng.next_f64() < sigmoid(steepness * z + offset)));
        out.outcome.push(1.0 + z + slope * z * treated + 0.5 * standard_normal(&mut rng));
        out.treatment.push(treated);
        out.z.push(z);
        for column in &mut out.noise {
            column.push(standard_normal(&mut rng));
        }
    }
    out
}

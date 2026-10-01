//! Known-truth structural models for the smoothed dose-response transport cell (2.2B X4).
//!
//! One baseline covariate `z` (graph coordinate 0), a randomized continuous dose `a`
//! (coordinate 1) and a continuous outcome `y` (coordinate 2); the selection diagram marks
//! `z`, so the certificate is a baseline standardization. The trial law of `z` is
//! `N(0, 1)`. The dose is randomized with the **known** conditional density
//!
//! ```text
//! pi(a | z) = (1 + beta(z) (a - 2) / 2) / 4  on [0, 4],   beta(z) = 0.6 tanh(z),
//! ```
//!
//! drawn exactly by rejection from the uniform law (acceptance `pi / 0.4`). The outcome is
//! `y = a^2 + z (1 + a / 2) + e`, `e ~ N(0, 1)` (the `Kinked` scenario instead uses
//! `y = a + 3 (a - 2.1)_+ + z + e`). For the quadratic outcome the smoothed target is, in
//! closed form,
//!
//! ```text
//! psi_h(a) = a^2 + h^2 / 5 + m_T (1 + a / 2),
//! ```
//!
//! because `integral K_h(a - t) t dt = a` and `integral K_h(a - t) t^2 dt = a^2 + h^2 / 5`
//! for the Epanechnikov kernel (second moment `1/5`), and `E_T[z] = m_T`. The point curve
//! is `psi_0(a) = a^2 + m_T (1 + a / 2)`.
//!
//! * `Good`: target `z ~ N(0.6, 1)`; both nuisance families correct with a degree-2
//!   dose basis with interactions and a linear logistic membership (equal variances).
//! * `Robust`: target `z ~ N(0.8, 1)` and the outcome `y = a^2 + z (1 + a/2) + z a^2 + e`,
//!   whose smoothed target is `psi_h(a) = (a^2 + h^2/5)(1 + m_T) + m_T (1 + a/2)`. A fit
//!   linear in the dose (degree-1 basis, interactions kept) misses `(1 + z)` times the
//!   curvature, an error that varies with `z`, so neither a constant odds weight nor a
//!   wrong dose density can repair it; the linear logistic membership is right.
//! * `RobustVarianceShift`: the `Robust` outcome with target `z ~ N(0.5, 1.4^2)`; the log
//!   sample odds are quadratic in `z`, outside a linear logistic model, so the fitted odds
//!   are wrong; the degree-2 basis with interactions is the right outcome family.
//! * `WeakOverlap`: target `z ~ N(3, 1)`; membership probabilities vanish in the tail.
//! * `Kinked`: target `z ~ N(0.6, 1)`, hinge outcome at `2.1`; `psi_h` is not needed, only
//!   the quadrature of the fitted hinge curve.
//!
//! [`Design::NestedCohort`] draws one IID cohort whose rows are trial participants with
//! probability `n_trial / (n_trial + n_target)`; [`Design::IndependentSamples`] draws
//! exactly `n_trial` trial rows then `n_target` target rows.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(dead_code)]

use antecedent_core::{SmoothedDoseTransportQuery, SmoothingKernel, VariableId};
use antecedent_estimate::{SmoothedDoseInput, TrialSampling};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Design {
    NestedCohort,
    IndependentSamples,
}

impl Design {
    pub const fn sampling(self) -> TrialSampling {
        match self {
            Self::NestedCohort => TrialSampling::NestedCohort,
            Self::IndependentSamples => TrialSampling::IndependentSamples,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scenario {
    Good,
    Robust,
    RobustVarianceShift,
    WeakOverlap,
    Kinked,
}

impl Scenario {
    /// `(mean, sd)` of the target covariate law.
    pub const fn target_law(self) -> (f64, f64) {
        match self {
            Self::Good | Self::Kinked => (0.6, 1.0),
            Self::Robust => (0.8, 1.0),
            Self::RobustVarianceShift => (0.5, 1.4),
            Self::WeakOverlap => (3.0, 1.0),
        }
    }

    /// The closed-form smoothed target `psi_h(a)` of the quadratic outcome (and of the
    /// `Robust` outcome, whose `z a^2` term adds `m_T (a^2 + h^2/5)`).
    pub fn truth(self, a: f64, h: f64) -> f64 {
        let (mean, _) = self.target_law();
        let curvature = match self {
            Self::Robust | Self::RobustVarianceShift => 1.0 + mean,
            _ => 1.0,
        };
        (a * a + h * h / 5.0) * curvature + mean * (1.0 + a / 2.0)
    }

    /// The point curve `psi_0(a)`.
    pub fn point_curve(self, a: f64) -> f64 {
        self.truth(a, 0.0)
    }
}

/// The known dose density of the design.
pub fn dose_density(dose: f64, z: f64) -> f64 {
    0.25 * (1.0 + 0.6 * z.tanh() * (dose - 2.0) / 2.0)
}

/// The mean outcome `mu(t, z)` of a scenario.
pub fn mean_outcome(scenario: Scenario, dose: f64, z: f64) -> f64 {
    match scenario {
        Scenario::Kinked => dose + 3.0 * (dose - 2.1).max(0.0) + z,
        Scenario::Robust | Scenario::RobustVarianceShift => {
            dose * dose + z * (1.0 + dose / 2.0) + z * dose * dose
        }
        _ => dose * dose + z * (1.0 + dose / 2.0),
    }
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

pub struct Stream(pub u64);

impl Stream {
    pub fn uniform(&mut self) -> f64 {
        self.0 = mix(self.0);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    /// An exact draw from `pi(. | z)` by rejection from the uniform law on `[0, 4]`.
    pub fn dose(&mut self, z: f64) -> f64 {
        loop {
            let candidate = 4.0 * self.uniform();
            if self.uniform() * 0.4 < dose_density(candidate, z) {
                return candidate;
            }
        }
    }
}

/// One dataset of a scenario under a design.
pub fn draw(
    design: Design,
    scenario: Scenario,
    n_trial: usize,
    n_target: usize,
    seed: u64,
) -> SmoothedDoseInput {
    let mut stream = Stream(mix(seed) ^ 0x05D0_5E00);
    let (mean, sd) = scenario.target_law();
    let total = n_trial + n_target;
    let participant: Vec<bool> = match design {
        Design::IndependentSamples => (0..total).map(|i| i < n_trial).collect(),
        Design::NestedCohort => {
            let p = n_trial as f64 / total as f64;
            (0..total).map(|_| stream.uniform() < p).collect()
        }
    };
    let (mut z, mut y, mut a, mut density) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for &in_trial in &participant {
        let covariate = if in_trial { stream.normal() } else { mean + sd * stream.normal() };
        let dose = stream.dose(covariate);
        let noise = stream.normal();
        z.push(covariate);
        if in_trial {
            a.push(dose);
            density.push(dose_density(dose, covariate));
            y.push(mean_outcome(scenario, dose, covariate) + noise);
        } else {
            a.push(0.0);
            density.push(0.0);
            y.push(0.0);
        }
    }
    SmoothedDoseInput {
        features: vec![0],
        covariates: vec![z],
        outcome: y,
        dose: a,
        dose_density: density,
        source: participant,
        sampling: design.sampling(),
    }
}

/// The selection diagram (`z` selected) and names of the DGP.
pub fn diagram() -> (SelectionDiagram, Vec<String>) {
    let mut graph = Admg::with_variables(3);
    for (from, to) in [(0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, vec![VariableId::from_raw(0)]).unwrap();
    (diagram, ["z", "a", "y"].map(String::from).to_vec())
}

/// The smoothed-dose query over `grid` with bandwidth `h` on the support `[0, 4]`.
pub fn query(grid: &[f64], h: f64) -> SmoothedDoseTransportQuery {
    SmoothedDoseTransportQuery {
        outcome: VariableId::from_raw(2),
        dose: VariableId::from_raw(1),
        source_population: Arc::from("trial"),
        target_population: Arc::from("target"),
        grid: Arc::from(grid),
        bandwidth: h,
        kernel: SmoothingKernel::Epanechnikov,
        dose_support: (0.0, 4.0),
        density_provenance: Arc::from("known"),
    }
}

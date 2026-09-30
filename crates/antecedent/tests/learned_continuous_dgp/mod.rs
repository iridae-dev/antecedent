//! Known-truth data-generating processes for the learned continuous transport cell (X4).
//!
//! One baseline covariate `z` (graph coordinate 0), a randomized binary treatment `a`
//! (coordinate 1, probability one half) and a continuous outcome `y` (coordinate 2).
//! The selection diagram marks `z`, so the certificate is a baseline standardization.
//! The trial law of `z` is `N(0, 1)`; scenarios change the target law and the outcome
//! model, and each has an exactly known target mean contrast.
//!
//! * `Good`: target `z ~ N(0.5, 1)`, `y = z + a (2 + z) + e`. The membership log-odds
//!   are linear in `z` and the outcome is linear, so both nuisance families are
//!   correctly specified. Truth `2.5`.
//! * `MisspecifiedOutcome`: target `z ~ N(0.7, 1)`, `y = z + a (1 + 3 z^2) + e`. The
//!   membership log-odds stay linear (equal variances), so the participation model is
//!   right, but the treatment effect is quadratic in `z`: a linear outcome learner is
//!   wrong, and its trial slope is exactly zero (`Cov(z, 1 + 3 z^2) = 0` under the
//!   symmetric trial law), so the outcome plug-in and the trial-only contrast both
//!   converge to `1 + 3 E_trial[z^2] = 4` while the truth is `1 + 3 (1 + 0.7^2) = 5.47`.
//!   A real target shift of `1.47` separates the double-robust estimator (which the
//!   theorem covers: participation model right) from either alternative.
//! * `MisspecifiedMembership`: target `z ~ N(0.5, 1.4^2)`, `y = z + a (2 + 2 z) + e`. The
//!   log sample odds are quadratic in `z`, outside a linear logistic model; the outcome is
//!   right. Truth `2 + 2 * 0.5 = 3`; the trial-only contrast (an estimator that ignores
//!   membership) converges to `2`, a real shift of `1`, and the theorem covers this
//!   case (outcome model right).
//! * Both families wrong at once is outside the theorem and is deliberately not a
//!   scenario: no claim is made or tested for it.
//! * `WeakOverlap`: target `z ~ N(3, 1)`; membership probabilities of the tail rows
//!   vanish, and the estimator must refuse rather than extrapolate.
//!
//! [`Design::NestedCohort`] draws one IID cohort whose rows are trial participants with
//! probability `n_trial / (n_trial + n_target)`; [`Design::IndependentSamples`] draws
//! exactly `n_trial` trial rows and `n_target` target rows.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(dead_code)]

use antecedent_core::{
    ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TransportQuery, VariableId,
};
use antecedent_estimate::{TrialAipwInput, TrialSampling};
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
    MisspecifiedOutcome,
    MisspecifiedMembership,
    WeakOverlap,
}

impl Scenario {
    /// The exactly known target mean contrast.
    pub fn truth(self) -> f64 {
        match self {
            Self::Good => 2.5,
            Self::MisspecifiedOutcome => 5.47,
            Self::MisspecifiedMembership => 3.0,
            Self::WeakOverlap => 5.0,
        }
    }

    /// The trial-only mean contrast: `E_trial[tau(z)]` under the `N(0, 1)` trial law. What an
    /// estimator that ignores source membership converges to.
    pub fn trial_only_contrast(self) -> f64 {
        match self {
            Self::MisspecifiedOutcome => 4.0,
            _ => 2.0,
        }
    }

    /// The limit of a linear-outcome plug-in over the target (no membership weighting).
    pub fn linear_plug_in(self) -> f64 {
        match self {
            Self::MisspecifiedOutcome => 4.0,
            other => other.truth(),
        }
    }

    fn target_law(self) -> (f64, f64) {
        match self {
            Self::Good => (0.5, 1.0),
            Self::MisspecifiedOutcome => (0.7, 1.0),
            Self::MisspecifiedMembership => (0.5, 1.4),
            Self::WeakOverlap => (3.0, 1.0),
        }
    }
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

struct Stream(u64);

impl Stream {
    fn uniform(&mut self) -> f64 {
        self.0 = mix(self.0);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// One dataset: the trial rows first (independent design), or a shuffled-by-draw cohort.
pub fn draw(
    design: Design,
    scenario: Scenario,
    n_trial: usize,
    n_target: usize,
    seed: u64,
) -> TrialAipwInput {
    let mut stream = Stream(mix(seed) ^ 0x51ED_2701);
    let (mean, sd) = scenario.target_law();
    let total = n_trial + n_target;
    let participant: Vec<bool> = match design {
        Design::IndependentSamples => (0..total).map(|i| i < n_trial).collect(),
        Design::NestedCohort => {
            let p = n_trial as f64 / total as f64;
            (0..total).map(|_| stream.uniform() < p).collect()
        }
    };
    let (mut z, mut y, mut a) = (Vec::new(), Vec::new(), Vec::new());
    for &in_trial in &participant {
        let covariate = if in_trial { stream.normal() } else { mean + sd * stream.normal() };
        let treated = stream.uniform() < 0.5;
        let noise = stream.normal();
        let arm = f64::from(u8::from(treated));
        let outcome = match scenario {
            Scenario::MisspecifiedOutcome => {
                covariate + arm * (1.0 + 3.0 * covariate * covariate) + noise
            }
            Scenario::MisspecifiedMembership => covariate + arm * (2.0 + 2.0 * covariate) + noise,
            _ => covariate + arm * (2.0 + covariate) + noise,
        };
        z.push(covariate);
        y.push(if in_trial { outcome } else { 0.0 });
        a.push(in_trial && treated);
    }
    TrialAipwInput {
        features: vec![0],
        covariates: vec![z],
        outcome: y,
        treatment: a,
        source: participant,
        randomization: vec![0.5; total],
        sampling: design.sampling(),
    }
}

/// The selection diagram (`z` selected), binary contrast query and names of the DGP.
pub fn graph() -> (SelectionDiagram, TransportQuery, Vec<String>) {
    let mut graph = Admg::with_variables(3);
    for (from, to) in [(0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, vec![VariableId::from_raw(0)]).unwrap();
    let query = TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(2),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(1),
                GridSpec::Values(Arc::from([0., 1.])),
            ),
        }),
        "trial",
        "target",
        [VariableId::from_raw(1)],
    );
    (diagram, query, ["z", "a", "y"].map(String::from).to_vec())
}

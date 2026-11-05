//! Whole-method coverage of the percentile bootstrap intervals of the nonlinear mediation
//! natural effects against the closed-form effects of the generating structural model.
//!
//! The procedure scored is `estimate_nonlinear_mediation` end to end (mediator regression,
//! polynomial outcome fit, Gauss-Hermite integration of the cross-world mediator law, and the
//! hand percentile interval of the retained row-bootstrap replicate effects). The interval's
//! construction equals the hand percentiles of the retained replicates in
//! `nonlinear_mediation_reference.rs`; this file measures whether it covers.
//!
//! Design and truth: the structural model of `nonlinear_mediation.rs`, with random draws instead
//! of the deterministic grid. `X` is uniform on `{-1, 0, 1, 2, 3}` (so `E[X] = 1`, `E[X^2] = 3`),
//! `A ~ Bernoulli(1/2)`, `M = 0.5 A + 0.5 X + e_M` with `e_M ~ N(0, 0.64)`, and
//! `Y = 1 + 0.8 A + 0.6 M + 0.4 M^2 + 0.3 A M + 0.7 X + e_Y` with `e_Y ~ N(0, 0.25)`. The outcome
//! polynomial of degree two with a treatment-by-mediator term is exactly this mean, so the model
//! the estimator fits is correct and the closed forms of that file hold:
//! `m1(a') = alpha a' + gamma E[X]`, `m2(a') = (alpha a')^2 + 2 alpha a' gamma E[X] +
//! gamma^2 E[X^2] + sigma2`, `theta(a, a') = b0 + b1 a + bx E[X] + (c1 + delta a) m1(a') +
//! c2 m2(a')`, `NDE = theta(1,0) - theta(0,0)`, `NIE = theta(1,1) - theta(1,0)`,
//! `TE = theta(1,1) - theta(0,0)`. Rows are on the three-point grid around 400 (200, 400, 800),
//! 200 bootstrap replicates (the estimator default), `n_sim()` datasets. A dataset the estimator
//! refuses (arm, overlap or support gates), or whose interval is withheld, is a skip and counts
//! as a miss.
//!
//! The three effects are three records of one test, labelled `direct`, `indirect` and `total`.
//! Each record is keyed by the calibration basis the estimate itself reports for that effect
//! (`calibration_basis`), the same key the runtime matches, so the record cannot describe
//! another construction.
//!
//! Assertion: `CoverageTally::assert`, the harness band around the nominal 0.95 for the first
//! `n_sim()` replicates, rechecked at 2000 near the edge.
//!
//! Ignored tests run via `scripts/measure_calibration_groups.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;

use antecedent_core::ExecutionContext;
use antecedent_estimate::nonlinear_mediation::{
    NonlinearMediationConfig, NonlinearMediationInput, NonlinearMediationPremises,
    estimate_nonlinear_mediation,
};
use calibration::{
    Construction, CoverageTally, RecordKey, ScopeFacts, gaussian, grid_n, grid_seed,
    map_replicates, n_sim, stream_seed, unit_uniform,
};

const BASE_ROWS: usize = 400;
const SEED_BASE: u64 = 0x4D3D_0001;
const ALPHA: f64 = 0.5;
const GAMMA: f64 = 0.5;
const SIGMA2: f64 = 0.64;
const B0: f64 = 1.0;
const B1: f64 = 0.8;
const C1: f64 = 0.6;
const C2: f64 = 0.4;
const DELTA: f64 = 0.3;
const BX: f64 = 0.7;
const EX: f64 = 1.0;
const EX2: f64 = 3.0;
const OUTCOME_SD: f64 = 0.5;
const XS: [f64; 5] = [-1.0, 0.0, 1.0, 2.0, 3.0];

fn m1(ap: f64) -> f64 {
    ALPHA * ap + GAMMA * EX
}

fn m2(ap: f64) -> f64 {
    (ALPHA * ap).powi(2) + 2.0 * ALPHA * ap * GAMMA * EX + GAMMA * GAMMA * EX2 + SIGMA2
}

fn theta(a: f64, ap: f64) -> f64 {
    B0 + B1 * a + BX * EX + (C1 + DELTA * a) * m1(ap) + C2 * m2(ap)
}

/// `(NDE, NIE, TE)` written out from the structural model, never from the estimator.
fn truth() -> [f64; 3] {
    [
        theta(1.0, 0.0) - theta(0.0, 0.0),
        theta(1.0, 1.0) - theta(1.0, 0.0),
        theta(1.0, 1.0) - theta(0.0, 0.0),
    ]
}

struct Draw {
    a: Vec<f64>,
    m: Vec<f64>,
    y: Vec<f64>,
    x: Vec<f64>,
}

fn draw(n: usize, seed: u64) -> Draw {
    let mut noise = gaussian(seed);
    let mut out = Draw { a: Vec::new(), m: Vec::new(), y: Vec::new(), x: Vec::new() };
    for i in 0..n {
        let uniform = |tag: u64| unit_uniform(stream_seed(seed, 2 * i as u64 + tag));
        let x = XS[((uniform(0) * 5.0) as usize).min(4)];
        let a = f64::from(uniform(1) < 0.5);
        let m = ALPHA * a + GAMMA * x + SIGMA2.sqrt() * noise();
        let y = B0 + B1 * a + C1 * m + C2 * m * m + DELTA * a * m + BX * x + OUTCOME_SD * noise();
        out.a.push(a);
        out.m.push(m);
        out.y.push(y);
        out.x.push(x);
    }
    out
}

/// The three effect intervals with their reported calibration bases, or `None` on a refusal or
/// a withheld interval.
type Scored = [((f64, f64), Construction, ScopeFacts, String); 3];

fn execute(n: usize, seed: u64) -> Option<Scored> {
    let data = draw(n, seed);
    let covariates = [data.x.as_slice()];
    let input = NonlinearMediationInput {
        treatment: &data.a,
        mediator: &data.m,
        outcome: &data.y,
        covariates: &covariates,
    };
    let estimate = estimate_nonlinear_mediation(
        &input,
        &NonlinearMediationPremises::sequentially_ignorable(),
        &NonlinearMediationConfig { seed, ..NonlinearMediationConfig::default() },
        &ExecutionContext::for_tests(seed),
    )
    .ok()?;
    let one = |effect: &str, interval: Option<(f64, f64)>| {
        let basis = estimate.calibration_basis(effect);
        Some((
            interval?,
            Construction {
                query: basis.query.to_string(),
                graph_class: basis.graph_class.to_string(),
                structure: basis.structure.to_string(),
                modality: basis.modality.to_string(),
                inference: basis.inference.to_string(),
                estimator: basis.estimator.to_string(),
                interval_method: basis.interval_method.to_string(),
                se_kind: basis.se_kind.to_string(),
                dependence: basis.dependence.to_string(),
                posterior: basis.posterior.to_string(),
                functional: basis.functional.to_string(),
                identification: basis.identification.to_string(),
                reported_level: basis.level,
            },
            ScopeFacts {
                row_count: basis.row_count,
                replicates_ok: basis.replicates_ok,
                posterior_draws: basis.posterior_draws,
                unidentified_mass: basis.unidentified_mass,
            },
            basis.validation.to_string(),
        ))
    };
    Some([
        one("natural_direct", estimate.natural_direct_interval)?,
        one("natural_indirect", estimate.natural_indirect_interval)?,
        one("total", estimate.total_interval)?,
    ])
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration_groups.sh"]
fn nonlinear_mediation_natural_effects_percentile_bootstrap_interval() {
    let n = grid_n(BASE_ROWS);
    let truths = truth();
    let new_tally = |label: &str| {
        CoverageTally::for_record(
            RecordKey {
                test: "nonlinear_mediation_natural_effects_percentile_bootstrap_interval",
                dgp: "gaussian_mediator_quadratic_outcome_closed_form_natural_effects",
                interval: "bootstrap_percentile",
            },
            calibration::REPORTED_LEVEL,
        )
        .labelled(label)
    };
    let mut tallies = [new_tally("direct"), new_tally("indirect"), new_tally("total")];
    let runs = map_replicates(n_sim(), |rep| execute(n, grid_seed(SEED_BASE + rep)));
    for run in runs {
        let Some(scored) = run else {
            for tally in &mut tallies {
                tally.skip();
            }
            continue;
        };
        for (k, (band, construction, scope, validation)) in scored.iter().enumerate() {
            tallies[k].bind_validation(construction, *scope, validation);
            tallies[k].record(Some(*band), truths[k]);
        }
    }
    for tally in &tallies {
        tally.assert();
    }
}

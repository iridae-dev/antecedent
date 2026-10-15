//! Whole-route calibration of the declared-cohort mean of a linear conditional effect.
//!
//! Nothing here is measured at implementation time: every test is `#[ignore]`d and runs
//! only at the final measurement step.
//!
//! The scored procedure is the whole route: draw rows, fit the effect natively (OLS of
//! the outcome on the declared basis and its treatment interaction, HC3 sandwich
//! covariance), export it with its retained `n` and `p`, bind a declared cohort, build
//! the decision source from the export and issue its Student-t interval. The target is
//! the declared cohort's mean of the true conditional effect `tau(x) = b' phi(x)`.
//!
//! Gated designs (nominal levels 0.90 and 0.95, sample size `grid_n(400)`):
//!
//! * `randomized_p3`: randomized treatment, two modifiers (`p = 3` basis terms).
//! * `confounded_p4`: treatment assigned from the observed modifiers, three modifiers.
//! * `confounded_p5`: the same with four modifiers.
//!
//! All use heteroskedastic noise (`sd = 0.5 exp(0.35 x_1)`), a declared cohort of 50 to
//! 200 rows drawn from the fitted rows, and uniform cohort weights.
//!
//! Named boundaries, each under its own calibration structure so that none shares a key
//! with a gated design:
//!
//! * `misspecified_effect`: the true effect has a quadratic component the linear basis
//!   cannot represent, so the cohort-mean target is biased.
//! * `high_leverage_cohort`: heavy-tailed modifiers and a cohort made of the rows with
//!   the largest `|x_1|`, where the sandwich covariance is least reliable.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "bounded indices and exact enumerated comparisons in a calibration fixture"
)]
#![cfg(feature = "calibration-internal")]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_arguments,
    reason = "bounded simulation sizes and small indices"
)]

#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "common/candidate_calibration.rs"]
mod candidate;

use antecedent_core::ExecutionContext;
use antecedent_estimate::cohort_linear_effect::{
    INTERVAL_METHOD, NativeCohortFit, boundary_calibration_basis, calibration_basis,
    native_cohort_interval_for,
};
use antecedent_io_core::cohort_decision::{
    CATE_FUNCTIONAL_ID, CohortBinding, CohortColumn, ContrastDeclaration,
};
use antecedent_io_core::quantity_wire::ScientificQuantityWire;
use calibration::{CoverageTally, RecordKey, grid_n, map_replicates, n_sim};
use candidate::Generator;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Boundary {
    None,
    MisspecifiedEffect,
    HighLeverageCohort,
}

impl Boundary {
    const fn label(self) -> &'static str {
        match self {
            Self::None => "gated",
            Self::MisspecifiedEffect => "misspecified_effect",
            Self::HighLeverageCohort => "high_leverage_cohort",
        }
    }
}

#[derive(Clone, Copy)]
struct Design {
    test: &'static str,
    dgp: &'static str,
    modifiers: usize,
    confounded: bool,
    boundary: Boundary,
    level: f64,
}

const EFFECT: [f64; 5] = [0.8, 0.5, -0.4, 0.3, -0.2];
const BASELINE: [f64; 4] = [0.4, -0.3, 0.2, -0.1];
const QUADRATIC_EFFECT: f64 = 0.6;

fn wire(id: &str, role: &str, functional: &str, regime: &str) -> ScientificQuantityWire {
    ScientificQuantityWire {
        version: 1,
        variable_id: id.into(),
        variable_name: id.into(),
        role: role.into(),
        units: "unit".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

struct Draw {
    features: Vec<Vec<f64>>,
    treatment: Vec<f64>,
    outcome: Vec<f64>,
    effect: Vec<f64>,
}

fn expit(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn draw(design: Design, n: usize, g: &mut Generator) -> Draw {
    let k = design.modifiers;
    let mut features = vec![Vec::with_capacity(n); k];
    let (mut treatment, mut outcome, mut effect) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let x: Vec<f64> = (0..k)
            .map(|j| {
                let z = g.normal();
                if design.boundary == Boundary::HighLeverageCohort && j == 0 {
                    z * z * z
                } else {
                    z
                }
            })
            .collect();
        let propensity = if design.confounded {
            expit(0.6 * x[0].clamp(-3.0, 3.0) - 0.4 * x.get(1).copied().unwrap_or(0.0))
        } else {
            0.5
        };
        let t = f64::from(u8::from(g.uniform() < propensity));
        let mut tau = EFFECT[0] + x.iter().zip(&EFFECT[1..]).map(|(a, b)| a * b).sum::<f64>();
        if design.boundary == Boundary::MisspecifiedEffect {
            tau += QUADRATIC_EFFECT * (x[0] * x[0] - 1.0);
        }
        let baseline = 1.0 + x.iter().zip(&BASELINE).map(|(a, b)| a * b).sum::<f64>();
        let sd = 0.5 * (0.35 * x[0].clamp(-3.0, 3.0)).exp();
        outcome.push(baseline + t * tau + sd * g.normal());
        treatment.push(t);
        effect.push(tau);
        for (column, value) in features.iter_mut().zip(x) {
            column.push(value);
        }
    }
    Draw { features, treatment, outcome, effect }
}

/// Declared cohort rows: a without-replacement sample of the fitted rows, or the rows
/// with the largest `|x_1|` for the high-leverage probe.
fn cohort_rows(design: Design, data: &Draw, size: usize, g: &mut Generator) -> Vec<usize> {
    let n = data.outcome.len();
    let mut index: Vec<usize> = (0..n).collect();
    if design.boundary == Boundary::HighLeverageCohort {
        index.sort_by(|&a, &b| data.features[0][b].abs().total_cmp(&data.features[0][a].abs()));
        index.truncate(size);
        return index;
    }
    for i in 0..size {
        let j = i + (g.uniform() * (n - i) as f64) as usize % (n - i);
        index.swap(i, j);
    }
    index.truncate(size);
    index
}

fn one_replicate(design: Design, n: usize, seed: u64) -> Option<(f64, f64, f64, f64)> {
    let mut g = Generator::new(seed);
    let data = draw(design, n, &mut g);
    let size = 50 + (g.uniform() * 151.0) as usize % 151;
    let rows = cohort_rows(design, &data, size.min(n / 2), &mut g);
    let columns: Vec<&[f64]> = data.features.iter().map(Vec::as_slice).collect();
    let regime = ContrastDeclaration::regime_of("t", 1.0, 0.0);
    let response = wire("y", "outcome", CATE_FUNCTIONAL_ID, &regime);
    let names: Vec<String> = (1..=design.modifiers).map(|j| format!("x{j}")).collect();
    let quantities: Vec<_> =
        names.iter().map(|name| wire(name, "covariate", "mean", "observed")).collect();
    let native =
        NativeCohortFit::fit(&data.outcome, &data.treatment, &columns, &response, &quantities)
            .ok()?;
    let cohort = CohortBinding::new(
        "calibration-cohort",
        "target",
        (0..rows.len()).map(|i| format!("r{i}")).collect(),
        names,
        data.features
            .iter()
            .map(|column| CohortColumn::Numeric(rows.iter().map(|&r| column[r]).collect()))
            .collect(),
        CohortBinding::uniform_weights(rows.len()).ok()?,
    )
    .ok()?;
    let (_, interval) = native_cohort_interval_for(
        &native,
        &cohort,
        "unit",
        design.level,
        &ExecutionContext::for_tests(seed),
    )
    .ok()?;
    let truth = rows.iter().map(|&r| data.effect[r]).sum::<f64>() / rows.len() as f64;
    Some((interval.lower, interval.upper, truth, interval.point))
}

fn run(design: Design, base: usize) {
    let n = grid_n(base);
    let mut tally = CoverageTally::for_record(
        RecordKey { test: design.test, dgp: design.dgp, interval: INTERVAL_METHOD },
        design.level,
    );
    let results = map_replicates(n_sim(), |replicate| {
        one_replicate(design, n, calibration::grid_seed(0x6c_0000 + replicate))
    });
    for result in results {
        let Some((lower, upper, truth, _)) = result else {
            tally.skip();
            continue;
        };
        let basis = match design.boundary {
            Boundary::None => calibration_basis(n as u64, design.level),
            other => boundary_calibration_basis(other.label(), n as u64, design.level),
        };
        candidate::bind(&mut tally, &basis);
        tally.record(Some((lower, upper)), truth);
    }
    match design.boundary {
        Boundary::None => tally.assert(),
        _ => tally.emit_named_boundary(),
    }
}

macro_rules! gated {
    ($name:ident, $dgp:literal, $k:expr, $confounded:expr, $level:expr) => {
        #[test]
        #[ignore = "calibration: run only at the final measurement step"]
        fn $name() {
            run(
                Design {
                    test: stringify!($name),
                    dgp: $dgp,
                    modifiers: $k,
                    confounded: $confounded,
                    boundary: Boundary::None,
                    level: $level,
                },
                400,
            );
        }
    };
}

gated!(
    cohort_linear_effect_randomized_p3_l90,
    "randomized_binary_treatment_two_modifiers_heteroskedastic_linear_effect_cohort_50_200",
    2,
    false,
    0.90
);
gated!(
    cohort_linear_effect_randomized_p3_l95,
    "randomized_binary_treatment_two_modifiers_heteroskedastic_linear_effect_cohort_50_200",
    2,
    false,
    0.95
);
gated!(
    cohort_linear_effect_confounded_p4_l90,
    "observed_confounded_binary_treatment_three_modifiers_heteroskedastic_linear_effect_cohort_50_200",
    3,
    true,
    0.90
);
gated!(
    cohort_linear_effect_confounded_p4_l95,
    "observed_confounded_binary_treatment_three_modifiers_heteroskedastic_linear_effect_cohort_50_200",
    3,
    true,
    0.95
);
gated!(
    cohort_linear_effect_confounded_p5_l90,
    "observed_confounded_binary_treatment_four_modifiers_heteroskedastic_linear_effect_cohort_50_200",
    4,
    true,
    0.90
);
gated!(
    cohort_linear_effect_confounded_p5_l95,
    "observed_confounded_binary_treatment_four_modifiers_heteroskedastic_linear_effect_cohort_50_200",
    4,
    true,
    0.95
);

macro_rules! boundary {
    ($name:ident, $dgp:literal, $boundary:expr, $level:expr) => {
        #[test]
        #[ignore = "calibration: run only at the final measurement step"]
        fn $name() {
            run(
                Design {
                    test: stringify!($name),
                    dgp: $dgp,
                    modifiers: 3,
                    confounded: true,
                    boundary: $boundary,
                    level: $level,
                },
                400,
            );
        }
    };
}

boundary!(
    cohort_linear_effect_misspecified_effect_boundary_l95,
    "observed_confounded_binary_treatment_quadratic_true_effect_linear_basis_cohort_50_200",
    Boundary::MisspecifiedEffect,
    0.95
);
boundary!(
    cohort_linear_effect_high_leverage_cohort_boundary_l95,
    "observed_confounded_binary_treatment_heavy_tailed_modifier_largest_leverage_cohort_50_200",
    Boundary::HighLeverageCohort,
    0.95
);

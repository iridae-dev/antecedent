//! Cross-fit sequential contrast on noisy two-decision data against an independent reference.
//!
//! The panel is generated from a fixed linear congruential stream with only additions and
//! products, so a numpy script written from the estimator's definition (logistic propensities,
//! per-fold least squares for the terminal and nested conditional means, the augmented
//! sequential score) reproduces the data bit for bit. Its values are the expectations below.
//! The generating effect of the always-treat regime over never-treat is `2.82` (first action
//! `1.0` plus `0.8 * 0.4` through the intermediate covariate plus second action `1.5`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, reason = "a 53-bit integer converts exactly")]

use antecedent_core::{
    ExecutionContext, NumericalPolicy, ProviderId, SemanticDigest, SourceDependencyId,
};
use antecedent_estimate::nuisance_cache::{
    CertifiedSource, FoldAssignment, FoldRow, NuisanceOptions, NumericalDependencies,
    NumericalDependency, ProviderDependency, TargetDependency,
};
use antecedent_estimate::score_posterior::ScorePosteriorOptions;
use antecedent_estimate::sequential_aipw_inference::{
    bayesian_sequential_contrast, sequential_contrast_wald_interval,
};
use antecedent_estimate::sequential_crossfit::{
    ActionLevel, MIN_PROBABILITY, SequentialIdentity, SequentialNuisanceStore, SequentialPanel,
    StaticRegime, estimate_sequential_contrast,
};

const SUBJECTS: usize = 80;

struct Panel {
    baseline: Vec<f64>,
    first: Vec<f64>,
    intermediate: Vec<f64>,
    second: Vec<f64>,
    outcome: Vec<f64>,
}

fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / 9_007_199_254_740_992.0
}

fn panel() -> Panel {
    let mut state = 12_345_u64;
    let mut out = Panel {
        baseline: Vec::new(),
        first: Vec::new(),
        intermediate: Vec::new(),
        second: Vec::new(),
        outcome: Vec::new(),
    };
    for _ in 0..SUBJECTS {
        let u1 = uniform(&mut state);
        let u2 = uniform(&mut state);
        let u3 = uniform(&mut state);
        let u4 = uniform(&mut state);
        let u5 = uniform(&mut state);
        let l0 = 2.0 * u1 - 1.0;
        let a0 = if u2 < 0.5 + 0.2 * l0 { 1.0 } else { 0.0 };
        let l1 = 0.5 * l0 + 0.8 * a0 + (2.0 * u3 - 1.0) * 0.5;
        let a1 = if u4 < 0.5 + 0.1 * (l1 - 0.4) { 1.0 } else { 0.0 };
        let y = 1.0 + 0.5 * l0 + 1.0 * a0 + 1.5 * a1 + 0.4 * l1 + (2.0 * u5 - 1.0);
        out.baseline.push(l0);
        out.first.push(a0);
        out.intermediate.push(l1);
        out.second.push(a1);
        out.outcome.push(y);
    }
    out
}

fn digest(mark: u8) -> SemanticDigest {
    let mut bytes = [0_u8; 32];
    bytes[31] = mark;
    SemanticDigest::from_bytes(bytes)
}

fn identity(panel: SequentialPanel<'_>) -> SequentialIdentity {
    let rows: Vec<FoldRow> = (0..SUBJECTS)
        .map(|row| FoldRow::new(row as u64, u32::from(matches!(row % 4, 1 | 2))))
        .collect();
    let floor = NumericalDependency::try_new(MIN_PROBABILITY, 0.05).expect("floor");
    SequentialIdentity::from_panel(
        panel,
        CertifiedSource::new(SourceDependencyId::from_raw(11), digest(3)),
        TargetDependency::new(digest(1)),
        FoldAssignment::try_new(2, &rows, &[]).expect("folds"),
        ProviderDependency::new(ProviderId::from_raw(4), digest(6)),
        NumericalDependencies::try_new(NumericalPolicy::strict(), &[floor]).expect("numerical"),
        &NuisanceOptions::new("noisy_reference"),
    )
    .expect("identity")
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() <= tolerance, "actual {actual} expected {expected}");
}

#[test]
fn noisy_contrast_matches_the_independent_reference() {
    let data = panel();
    let baseline = [data.baseline.as_slice()];
    let intermediate = [data.intermediate.as_slice()];
    let view = SequentialPanel {
        baseline: &baseline,
        first_treatment: &data.first,
        intermediate: &intermediate,
        second_treatment: &data.second,
        outcome: &data.outcome,
    };
    let treated = StaticRegime::new(ActionLevel::One, ActionLevel::One);
    let untreated = StaticRegime::new(ActionLevel::Zero, ActionLevel::Zero);
    let mut store = SequentialNuisanceStore::new();
    let result =
        estimate_sequential_contrast(view, &identity(view), treated, untreated, &mut store)
            .expect("contrast");

    close(result.active_mean, 3.577_003_395_896_987_6, 1.0e-4);
    close(result.control_mean, 1.346_150_090_756_641_5, 1.0e-4);
    close(result.contrast, 2.230_853_305_140_346, 1.0e-4);
    close(result.unmeasured_score_scale, 0.689_242_100_473_076_1, 1.0e-4);
    assert_eq!(result.contrast_scores.len(), SUBJECTS);

    let interval = sequential_contrast_wald_interval(&result, 0.95).expect("interval");
    let half = 1.959_963_984_540_054 * 0.689_242_100_473_076_1;
    close(interval.interval.0, 2.230_853_305_140_346 - half, 1.0e-3);
    close(interval.interval.1, 2.230_853_305_140_346 + half, 1.0e-3);

    let ctx = ExecutionContext::for_tests(41);
    let options = ScorePosteriorOptions { draws: 4_000, coverage: 0.95 };
    let posterior = bayesian_sequential_contrast(&result, options, &ctx).expect("posterior");
    close(posterior.contrast.score_mean, result.contrast, 1.0e-12);
    // The Bayesian-bootstrap spread is the Dirichlet moment of the cross-fit scores, which is
    // the score scale up to the (n - 1) / (n + 1) factor of the two divisors.
    let n = SUBJECTS as f64;
    let expected = result.unmeasured_score_scale * ((n - 1.0) / (n + 1.0)).sqrt();
    close(posterior.contrast.standard_deviation, expected, 0.06 * expected);
}

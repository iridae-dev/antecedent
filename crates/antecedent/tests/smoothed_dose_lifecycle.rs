//! Smoothed dose-response transport grid (2.2B cell X4): known-truth execution under both
//! sampling designs, exact-integration and oracle-score checks of the derivation, the
//! smoothing-bias diagnostic, one-family misspecification, support and request refusals,
//! the quadrature tolerance, invariances, the out-of-fold identity, the estimator menu,
//! the prepared lifecycle, the independently replayed artifact and the closed interval
//! route.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    reason = "integration test: exact replays compare bits and fixture counts are small"
)]

#[path = "smoothed_dose_dgp/mod.rs"]
mod dgp;

use antecedent::{PreparedSmoothedDose, StudyBuilder, consume_smoothed_dose_artifact};
use antecedent_core::{
    ExecutionContext, NonZeroThreadCount, Parallelism, SmoothedDoseTransportQuery,
};
use antecedent_estimate::smoothed_dose::{
    DoseBasis, SmoothedDoseEstimate, smoothed_dose_bootstrap_groups, smoothed_dose_fold_assignment,
};
use antecedent_estimate::{
    EstimationError, LearnerSpec, LinearSpec, SmoothedDoseInput, SmoothedDoseOptions,
    smoothed_dose_interval_internal,
};
use antecedent_identify::{TransportIdentification, TransportIdentifier};
use antecedent_io::IoError;
use antecedent_io::smoothed_dose_artifact::{
    SmoothedDoseArtifactError as Refusal, SmoothedDoseArtifactWire, SmoothedDoseConsumeLimits,
};
use antecedent_learn::PredictionMap;
use dgp::{Design, Scenario, diagram, draw, query};

const DESIGNS: [Design; 2] = [Design::NestedCohort, Design::IndependentSamples];
const GRID: [f64; 3] = [1.0, 2.0, 3.0];
const H: f64 = 0.5;

fn options() -> SmoothedDoseOptions {
    SmoothedDoseOptions { folds: 3, ..SmoothedDoseOptions::default() }
}

fn identify(query: &SmoothedDoseTransportQuery) -> TransportIdentification {
    let (diagram, _) = diagram();
    TransportIdentifier::new().identify(&diagram, &query.transport_query()).unwrap()
}

fn prepare(
    query: SmoothedDoseTransportQuery,
    input: SmoothedDoseInput,
    options: SmoothedDoseOptions,
    seed: u64,
) -> Result<PreparedSmoothedDose, IoError> {
    let (diagram, names) = diagram();
    StudyBuilder::smoothed_dose_transport(
        diagram,
        query,
        input,
        options,
        names,
        &ExecutionContext::for_tests(seed),
    )
}

fn refused_code(error: IoError) -> (&'static str, String) {
    match error {
        IoError::Refused { code, message } => (code, message),
        other => panic!("expected a coded refusal, got {other}"),
    }
}

fn estimation_refusal(error: &EstimationError) -> (&'static str, &str) {
    match error {
        EstimationError::Refused { code, message } => (code, message.as_str()),
        other => panic!("expected a coded refusal, got {other:?}"),
    }
}

fn estimate(
    design: Design,
    scenario: Scenario,
    grid: &[f64],
    h: f64,
    n: (usize, usize),
    options: &SmoothedDoseOptions,
    seed: u64,
) -> Result<SmoothedDoseEstimate, EstimationError> {
    let q = query(grid, h);
    let input = draw(design, scenario, n.0, n.1, seed);
    antecedent_estimate::estimate_smoothed_dose(
        &identify(&q),
        &q,
        &input,
        options,
        &ExecutionContext::for_tests(seed),
    )
}

#[test]
fn known_truth_curve_under_both_designs() {
    for design in DESIGNS {
        let builder_query = query(&GRID, H);
        let builder_rows = draw(design, Scenario::Good, 4000, 3000, 11);
        let prepared = prepare(builder_query, builder_rows, options(), 11).unwrap();
        // The prepared plan owns its rows and certificate; nothing else is kept.
        let plan = prepared.estimate(&ExecutionContext::for_tests(11)).unwrap();
        let result = plan.estimate();
        assert_eq!(result.grid.len(), GRID.len());
        for point in &result.grid {
            let truth = Scenario::Good.truth(point.dose, H);
            assert!(
                (point.estimate - truth).abs() < 0.2,
                "{design:?} a={}: {} vs {truth}",
                point.dose,
                point.estimate
            );
            // Numerical error, smoothing bias and sampling diagnostics are separate records.
            assert!(point.quadrature.estimate_error < 1e-9, "{:?}", point.quadrature);
            assert_eq!((point.quadrature.nodes, point.quadrature.check_nodes), (16, 32));
            assert!(point.influence_se_diagnostic > 0.0 && point.influence_se_diagnostic < 0.2);
            assert!(point.support.effective_sample_size > 100.0);
            assert!((point.plug_in + point.augmentation - point.estimate).abs() < 1e-12);
        }
        assert_eq!(result.uncertainty.status, "point_only");
        assert!(!result.uncertainty.available());
        // Provider provenance for every fitted nuisance, the fold scheme and overlap.
        assert_eq!(result.provenance.len(), 2 * 3);
        assert!(result.provenance.iter().all(|p| !p.implementation.is_empty()));
        assert_eq!(result.folds.scheme, "stratified_round_robin_source_and_target");
        assert_eq!(result.folds.assignment.len(), 7000);
        assert!(result.overlap.probability_min > 0.05 && result.overlap.max_odds_weight > 1.0);
        assert!(
            result.diagnostics.outcome_rmse < 1.2 && result.diagnostics.membership_logloss > 0.0
        );
    }
}

/// `E_K[(a + h u)^k]` for the Epanechnikov kernel: the moments `1, 0, 1/5, 0` of `u`.
fn kernel_power(a: f64, h: f64, k: usize) -> f64 {
    match k {
        0 => 1.0,
        1 => a,
        2 => a * a + h * h / 5.0,
        3 => a.powi(3) + 3.0 * a * h * h / 5.0,
        _ => unreachable!(),
    }
}

/// `E_K[(a + h u - c)_+]` in closed form: `h * 0.75 (1/4 - 2d/3 + d^2/2 - d^4/12)` with
/// `d = (c - a)/h` clipped to `[-1, 1]`.
fn kernel_hinge(a: f64, h: f64, c: f64) -> f64 {
    let d = ((c - a) / h).clamp(-1.0, 1.0);
    let base = h * 0.75 * (0.25 - 2.0 * d / 3.0 + d * d / 2.0 - d.powi(4) / 12.0);
    // For d < -1 the hinge is linear on the whole window: a - c exactly.
    if (c - a) / h < -1.0 { a - c } else { base }
}

/// The exact kernel integral of a stored linear outcome model at covariate `z`.
fn exact_nu(coefficients: &[f64], basis: &DoseBasis, a: f64, h: f64, z: f64) -> f64 {
    let degree = usize::from(basis.degree);
    let dose_terms: Vec<f64> = (1..=degree)
        .map(|k| kernel_power(a, h, k))
        .chain(basis.knots.iter().map(|c| kernel_hinge(a, h, *c)))
        .collect();
    let mut value = coefficients[0];
    for (d, m) in dose_terms.iter().enumerate() {
        value += coefficients[1 + d] * m;
    }
    value += coefficients[1 + dose_terms.len()] * z;
    if basis.interactions {
        for (d, m) in dose_terms.iter().enumerate() {
            value += coefficients[2 + dose_terms.len() + d] * m * z;
        }
    }
    value
}

fn exact_plug_in(
    result: &SmoothedDoseEstimate,
    input: &SmoothedDoseInput,
    basis: &DoseBasis,
    a: f64,
    h: f64,
) -> f64 {
    let target: Vec<usize> = (0..input.source.len()).filter(|i| !input.source[*i]).collect();
    let mut total = 0.0;
    for &i in &target {
        let fold = usize::from(result.folds.assignment[i]);
        let PredictionMap::Linear { coefficients, .. } = &result.models.outcome[fold].model else {
            panic!("a linear outcome model");
        };
        total += exact_nu(coefficients, basis, a, h, input.covariates[0][i]);
    }
    total / target.len() as f64
}

#[test]
fn quadrature_matches_exact_integration_and_the_oracle_score_is_centred() {
    // (a) The closed form psi_h agrees with independent numerical integration of the SCM:
    // a midpoint rule in the dose and Gauss-Hermite in the target covariate.
    let (nodes, weights) = antecedent_stats::special::gauss_hermite_standard_normal(20);
    for a in GRID {
        for h in [0.25, 0.5, 1.0] {
            let steps = 20_000;
            let mut numeric = 0.0;
            for (x, w) in nodes.iter().zip(&weights) {
                let z = 0.6 + x;
                let mut inner = 0.0;
                for s in 0..steps {
                    let u = -1.0 + (f64::from(s) + 0.5) * 2.0 / f64::from(steps);
                    inner += 0.75 * (1.0 - u * u) * dgp::mean_outcome(Scenario::Good, a + h * u, z);
                }
                numeric += w * inner * 2.0 / f64::from(steps);
            }
            let closed = Scenario::Good.truth(a, h);
            assert!((numeric - closed).abs() < 1e-7, "a={a} h={h}: {numeric} vs {closed}");
        }
    }
    // (b) The estimator's quadrature of every stored fold model equals the model's exact
    // kernel integral (Gauss-Legendre with 32 nodes is exact for this polynomial basis).
    let input = draw(Design::IndependentSamples, Scenario::Good, 2000, 1500, 3);
    for basis in [DoseBasis::default(), DoseBasis { degree: 3, knots: vec![], interactions: true }]
    {
        let opts = SmoothedDoseOptions { basis: basis.clone(), ..options() };
        let q = query(&GRID, H);
        let result = antecedent_estimate::estimate_smoothed_dose(
            &identify(&q),
            &q,
            &input,
            &opts,
            &ExecutionContext::for_tests(3),
        )
        .unwrap();
        for point in &result.grid {
            let exact = exact_plug_in(&result, &input, &basis, point.dose, H);
            assert!(
                (point.plug_in - exact).abs() < 1e-10,
                "{}: {} vs {exact}",
                point.dose,
                point.plug_in
            );
            let half = exact_plug_in(&result, &input, &basis, point.dose, H / 2.0);
            assert!(
                (point.smoothing_bias.half_bandwidth_difference - (exact - half)).abs() < 1e-10
            );
        }
    }
    // (c) The oracle score - true mu, true membership odds, known density - is centred on
    // psi_h: the composed score has mean zero at the truth (independent samples, where
    // P(S=1 | z) = n1 phi(z) / (n1 phi(z) + n0 phi(z - m))).
    let (n1, n0) = (40_000usize, 30_000usize);
    let big = draw(Design::IndependentSamples, Scenario::Good, n1, n0, 17);
    let (m, _) = Scenario::Good.target_law();
    for a in GRID {
        let mut total = 0.0;
        let mut squares = Vec::new();
        for i in 0..big.source.len() {
            let z = big.covariates[0][i];
            if big.source[i] {
                let ratio = (n0 as f64 / n1 as f64) * (m * z - m * m / 2.0).exp();
                let dose = big.dose[i];
                let u = (dose - a) / H;
                let k = if u.abs() <= 1.0 { 0.75 * (1.0 - u * u) / H } else { 0.0 };
                let term = ratio * k / big.dose_density[i]
                    * (big.outcome[i] - dgp::mean_outcome(Scenario::Good, dose, z));
                total += term;
                squares.push(term);
            } else {
                let nu = a * a + H * H / 5.0 + z * (1.0 + a / 2.0);
                total += nu;
                squares.push(nu - Scenario::Good.truth(a, H));
            }
        }
        let oracle = total / n0 as f64;
        let se = squares.iter().map(|t| t * t).sum::<f64>().sqrt() / n0 as f64;
        let truth = Scenario::Good.truth(a, H);
        assert!((oracle - truth).abs() < 4.0 * se, "a={a}: {oracle} vs {truth} (se {se})");
        assert!(se < 0.05);
    }
}

#[test]
fn smoothing_bias_is_reported_apart_and_the_estimate_targets_psi_h() {
    let h = 1.2;
    for design in DESIGNS {
        let result =
            estimate(design, Scenario::Good, &[2.0], h, (8000, 6000), &options(), 23).unwrap();
        let point = &result.grid[0];
        let psi_h = Scenario::Good.truth(2.0, h);
        let psi_0 = Scenario::Good.point_curve(2.0);
        // The gap psi_h - psi_0 = h^2/5 = 0.288 is detected: the estimate is psi_h's.
        assert!((point.estimate - psi_h).abs() < 0.08, "{design:?}: {} vs {psi_h}", point.estimate);
        assert!((point.estimate - psi_0).abs() > 0.2, "{design:?}: {} vs {psi_0}", point.estimate);
        // The plug-in half-bandwidth difference matches its closed form 3 h^2 / 20 for a
        // curve quadratic in the dose, and 4/3 of it recovers h^2 / 5.
        let bias = &point.smoothing_bias;
        assert!((bias.half_bandwidth_difference - 3.0 * h * h / 20.0).abs() < 0.03, "{bias:?}");
        assert!((bias.local_quadratic_bias - h * h / 5.0).abs() < 0.04, "{bias:?}");
        assert!(
            (bias.local_quadratic_bias - 4.0 / 3.0 * bias.half_bandwidth_difference).abs() < 1e-12
        );
        // The diagnostic is never added: the estimate is plug-in plus augmentation only.
        assert_eq!(point.estimate, point.plug_in + point.augmentation);
    }
}

/// The trial-only kernel-weighted comparator: `mean_trial K_h(a - A) Y / pi`, an estimator
/// that ignores source membership (it converges to the trial population's `psi_h`).
fn trial_only(input: &SmoothedDoseInput, a: f64, h: f64) -> f64 {
    let rows: Vec<usize> = (0..input.source.len()).filter(|i| input.source[*i]).collect();
    rows.iter()
        .map(|&i| {
            let u = (input.dose[i] - a) / h;
            let k = if u.abs() <= 1.0 { 0.75 * (1.0 - u * u) / h } else { 0.0 };
            k * input.outcome[i] / input.dose_density[i]
        })
        .sum::<f64>()
        / rows.len() as f64
}

#[test]
fn misspecified_nuisance_cases_match_only_the_claimed_robustness() {
    // Model double robustness: one nuisance family wrong, the other right, each with a
    // real target shift. Both wrong at once is outside the claim and is not asserted.
    const TOLERANCE: f64 = 0.35;
    for design in DESIGNS {
        // Outcome family wrong (linear in the dose; the truth is quadratic), membership
        // right. The outcome plug-in misses the curvature over the window (about 1.28 at
        // a = 2) and the trial-only comparator misses the covariate shift (0.7 * 2).
        let scenario = Scenario::ShiftedFar;
        let wrong_outcome = SmoothedDoseOptions {
            basis: DoseBasis { degree: 1, knots: vec![], interactions: true },
            min_membership_probability: 0.02,
            ..options()
        };
        let input = draw(design, scenario, 4000, 3000, 5);
        let result =
            estimate(design, scenario, &[2.0], H, (4000, 3000), &wrong_outcome, 5).unwrap();
        let point = &result.grid[0];
        let truth = scenario.truth(2.0, H);
        assert!(
            (point.estimate - truth).abs() < TOLERANCE,
            "{design:?}: {} vs {truth}",
            point.estimate
        );
        let trial = trial_only(&input, 2.0, H);
        for (name, alternative) in [("plug-in", point.plug_in), ("trial-only", trial)] {
            assert!(
                (alternative - truth).abs() > 2.5 * TOLERANCE,
                "{design:?}: the {name} alternative {alternative} is not visibly wrong ({truth})"
            );
        }
        assert!((trial - Scenario::trial_only(2.0, H)).abs() < 0.4, "{design:?}: {trial}");

        // Membership family wrong (variance-shifted target: quadratic log odds against a
        // linear logistic), outcome right. Ignoring membership misses 0.5 (1 + a/2).
        let scenario = Scenario::VarianceShift;
        let input = draw(design, scenario, 4000, 3000, 6);
        let result =
            estimate(design, scenario, &[2.5, 3.0], H, (4000, 3000), &options(), 6).unwrap();
        for point in &result.grid {
            let truth = scenario.truth(point.dose, H);
            assert!(
                (point.estimate - truth).abs() < 0.3,
                "{design:?} a={}: {} vs {truth}",
                point.dose,
                point.estimate
            );
            let trial = trial_only(&input, point.dose, H);
            assert!(
                (trial - truth).abs() > 2.5 * 0.3,
                "{design:?}: ignoring membership ({trial}) is not visibly wrong ({truth})"
            );
        }
    }
}

#[test]
fn weak_overlap_and_thin_dose_support_refuse_rather_than_extrapolate() {
    for design in DESIGNS {
        let error = estimate(design, Scenario::WeakOverlap, &GRID, H, (1500, 1000), &options(), 3)
            .unwrap_err();
        let (code, message) = estimation_refusal(&error);
        assert_eq!(code, "transport_support_failure");
        assert!(message.starts_with("dose_response.membership_overlap"), "{message}");
    }
    let base = draw(Design::IndependentSamples, Scenario::Good, 1500, 1000, 4);
    let q = query(&GRID, H);
    let id = identify(&q);
    let check = |input: &SmoothedDoseInput, options: &SmoothedDoseOptions| {
        let error =
            antecedent_estimate::validate_smoothed_dose(&id, &q, input, options).unwrap_err();
        let (code, message) = estimation_refusal(&error);
        (code, message.split(':').next().unwrap().to_owned())
    };
    assert!(antecedent_estimate::validate_smoothed_dose(&id, &q, &base, &options()).is_ok());
    // A known density below the declared floor, then invalid densities and doses.
    let mut floored = base.clone();
    floored.dose_density[0] = 1e-4;
    assert_eq!(
        check(&floored, &options()),
        ("transport_support_failure", "dose_response.dose_density_floor".into())
    );
    for edit in [
        |i: &mut SmoothedDoseInput| i.dose_density[1] = 0.0,
        |i: &mut SmoothedDoseInput| i.dose_density[1] = f64::NAN,
        |i: &mut SmoothedDoseInput| i.dose[1] = 4.5,
    ] {
        let mut invalid = base.clone();
        edit(&mut invalid);
        assert_eq!(
            check(&invalid, &options()),
            ("transport_support_failure", "dose_response.dose_density_invalid".into())
        );
    }
    // Thin local support: move all but eight source doses out of the window at a = 2.
    let mut thin = base.clone();
    let mut kept = 0;
    for i in 0..thin.source.len() {
        if thin.source[i] && (thin.dose[i] - 2.0).abs() < H {
            if kept < 8 {
                kept += 1;
            } else {
                thin.dose[i] = if thin.dose[i] < 2.0 { thin.dose[i] - H } else { thin.dose[i] + H };
            }
        }
    }
    let loose = SmoothedDoseOptions { min_distinct_doses: 2, ..options() };
    assert_eq!(
        check(&thin, &loose),
        ("transport_support_failure", "dose_response.local_dose_ess".into())
    );
    // A too-discrete dose: rounded to integers, one dose value carries all weight at 2.
    let mut discrete = base;
    for i in 0..discrete.source.len() {
        if discrete.source[i] {
            discrete.dose[i] = discrete.dose[i].round();
        }
    }
    assert_eq!(
        check(&discrete, &options()),
        ("treatment_support_too_discrete", "dose_response.dose_too_discrete".into())
    );
}

#[test]
#[allow(clippy::too_many_lines, reason = "one refusal per assertion group")]
fn requests_outside_the_cell_refuse_with_their_details() {
    let base = draw(Design::IndependentSamples, Scenario::Good, 1500, 1000, 8);
    let detail = |error: IoError| {
        let (code, message) = refused_code(error);
        (code, message.split(':').next().unwrap().to_owned())
    };
    // An extrapolative grid window (no boundary kernels).
    assert_eq!(
        detail(prepare(query(&[3.8], H), base.clone(), options(), 1).unwrap_err()),
        ("transport_support_failure", "dose_response.grid_outside_dose_support".into())
    );
    assert_eq!(
        detail(prepare(query(&[0.4], H), base.clone(), options(), 1).unwrap_err()).1,
        "dose_response.grid_outside_dose_support"
    );
    // An estimated density.
    let mut estimated = query(&GRID, H);
    estimated.density_provenance = "estimated".into();
    assert_eq!(
        detail(prepare(estimated, base.clone(), options(), 1).unwrap_err()),
        ("route_not_supported", "dose_response.estimated_dose_density".into())
    );
    // A non-portable learner.
    let neural = SmoothedDoseOptions {
        outcome: LearnerSpec::NeuralNet(antecedent_estimate::NeuralSpec::default()),
        ..options()
    };
    assert_eq!(
        detail(prepare(query(&GRID, H), base.clone(), neural, 1).unwrap_err()),
        ("route_not_supported", "dose_response.learner_not_portable".into())
    );
    // Named targets, sampling designs and kernels.
    let parse = |r: Result<(), EstimationError>| {
        let e = r.unwrap_err();
        let (code, message) = estimation_refusal(&e);
        (code, message.split(':').next().unwrap().to_owned())
    };
    for target in ["point_curve", "stochastic", "coarsened", "incremental", "derivative"] {
        assert_eq!(
            parse(antecedent_estimate::smoothed_dose::parse_smoothed_dose_target(target)),
            ("route_not_supported", "dose_response.target_not_smoothed".into())
        );
    }
    for target in ["cate", "simultaneous_band"] {
        assert_eq!(
            parse(antecedent_estimate::smoothed_dose::parse_smoothed_dose_target(target)),
            ("route_not_supported", "dose_response.cate_or_simultaneous".into())
        );
    }
    assert!(
        antecedent_estimate::smoothed_dose::parse_smoothed_dose_target("smoothed_dose_response")
            .is_ok()
    );
    assert_eq!(
        parse(
            antecedent_estimate::smoothed_dose::parse_smoothed_dose_sampling("clustered")
                .map(|_| ())
        ),
        ("sampling_dependence_unknown", "dose_response.non_iid_design".into())
    );
    assert_eq!(
        parse(antecedent_estimate::smoothed_dose::parse_smoothing_kernel("gaussian").map(|_| ())),
        ("route_not_supported", "dose_response.kernel_not_supported".into())
    );
    // Every bound at its cap is accepted and at cap + 1 refused.
    let q = query(&GRID, H);
    let id = identify(&q);
    let validate =
        |q: &SmoothedDoseTransportQuery, input: &SmoothedDoseInput, o: &SmoothedDoseOptions| {
            antecedent_estimate::validate_smoothed_dose(&id, q, input, o)
        };
    let bounds_refused = |r: Result<(), EstimationError>| {
        matches!(&r, Err(EstimationError::Refused { code: "route_not_supported", message })
            if message.starts_with("dose_response.bounds_exceeded"))
    };
    let grid16: Vec<f64> = (0..16).map(|i| 0.5 + 3.0 * f64::from(i) / 15.0).collect();
    let grid17: Vec<f64> = (0..17).map(|i| 0.5 + 3.0 * f64::from(i) / 16.0).collect();
    assert!(validate(&query(&grid16, H), &base, &options()).is_ok());
    assert!(bounds_refused(validate(&query(&grid17, H), &base, &options())));
    for (cap, over) in [
        (
            SmoothedDoseOptions { folds: 20, ..options() },
            SmoothedDoseOptions { folds: 21, ..options() },
        ),
        (
            SmoothedDoseOptions { bootstrap: 2000, ..options() },
            SmoothedDoseOptions { bootstrap: 2001, ..options() },
        ),
        (
            SmoothedDoseOptions { quadrature_nodes: 32, ..options() },
            SmoothedDoseOptions { quadrature_nodes: 64, ..options() },
        ),
        (
            SmoothedDoseOptions {
                basis: DoseBasis { degree: 3, ..DoseBasis::default() },
                ..options()
            },
            SmoothedDoseOptions {
                basis: DoseBasis { degree: 4, ..DoseBasis::default() },
                ..options()
            },
        ),
        (
            SmoothedDoseOptions {
                basis: DoseBasis {
                    knots: (1..=8).map(|k| 0.4 * f64::from(k)).collect(),
                    ..DoseBasis::default()
                },
                ..options()
            },
            SmoothedDoseOptions {
                basis: DoseBasis {
                    knots: (1..=9).map(|k| 0.4 * f64::from(k)).collect(),
                    ..DoseBasis::default()
                },
                ..options()
            },
        ),
    ] {
        assert!(validate(&q, &base, &cap).is_ok(), "{cap:?}");
        assert!(bounds_refused(validate(&q, &base, &over)), "{over:?}");
    }
    assert!(bounds_refused(validate(
        &q,
        &base,
        &SmoothedDoseOptions { quadrature_nodes: 8, ..options() }
    )));
    // Rows: 200000 accepted, 200001 refused before anything else is read.
    let rows = |n: usize| {
        let mut big = draw(Design::IndependentSamples, Scenario::Good, n - n / 3, n / 3, 9);
        big.sampling = antecedent_estimate::TrialSampling::IndependentSamples;
        big
    };
    assert!(validate(&q, &rows(200_000), &options()).is_ok());
    assert!(bounds_refused(validate(&q, &rows(200_001), &options())));
    // Covariates: 256 pass the bound (and then fail the certificate), 257 are refused by it.
    let widen = |k: usize| {
        let mut wide = base.clone();
        wide.features = (0..k as u32).collect();
        wide.covariates = vec![base.covariates[0].clone(); k];
        wide
    };
    assert!(!bounds_refused(validate(&q, &widen(256), &options())));
    assert!(bounds_refused(validate(&q, &widen(257), &options())));
    // A replicate request below the floor keeps the point and withholds the interval.
    let below = SmoothedDoseOptions { bootstrap: 50, ..options() };
    let result = prepare(query(&GRID, H), base.clone(), below.clone(), 2)
        .unwrap()
        .estimate(&ExecutionContext::for_tests(2))
        .unwrap();
    let uncertainty = &result.estimate().uncertainty;
    assert_eq!(uncertainty.reason, "estimator_inference_mismatch");
    assert_eq!(uncertainty.detail.as_deref(), Some("dose_response.bootstrap_below_floor"));
    let error =
        smoothed_dose_interval_internal(&id, &q, &base, &below, &ExecutionContext::for_tests(2))
            .unwrap_err();
    assert_eq!(estimation_refusal(&error).0, "estimator_inference_mismatch");
}

#[test]
fn a_kinked_fitted_curve_surfaces_its_quadrature_error_and_refuses_at_a_tight_tolerance() {
    let input = draw(Design::IndependentSamples, Scenario::Kinked, 3000, 2000, 31);
    let q = query(&[2.0], H);
    let id = identify(&q);
    let hinge = DoseBasis { degree: 1, knots: vec![2.1], interactions: false };
    let run = |nodes: usize, tolerance: f64| {
        let opts = SmoothedDoseOptions {
            basis: hinge.clone(),
            quadrature_nodes: nodes,
            quadrature_tolerance: tolerance,
            ..options()
        };
        antecedent_estimate::estimate_smoothed_dose(
            &id,
            &q,
            &input,
            &opts,
            &ExecutionContext::for_tests(31),
        )
    };
    // A tight tolerance refuses: the kink inside the window defeats the 16-node rule.
    let error = run(16, 1e-6).unwrap_err();
    let (code, message) = estimation_refusal(&error);
    assert_eq!(code, "transport_numerical_failure");
    assert!(message.starts_with("dose_response.quadrature_tolerance"), "{message}");
    // A loose one passes and records the error, separate from the point.
    let coarse = run(16, 1e-2).unwrap();
    let fine = run(32, 1e-2).unwrap();
    let (c, f) = (&coarse.grid[0], &fine.grid[0]);
    assert!(c.quadrature.estimate_error > 1e-5, "{:?}", c.quadrature);
    // The error shrinks as Q grows, and it tracks the exact integration error of the
    // coarse rule on the fitted hinge curve (closed-form hinge moment).
    assert!(
        f.quadrature.estimate_error < c.quadrature.estimate_error,
        "{:?} {:?}",
        c.quadrature,
        f.quadrature
    );
    // Both runs fit the same models (same rows, folds and seed); only the rules differ.
    assert_eq!(coarse.models, fine.models);
    let exact = exact_plug_in(&coarse, &input, &hinge, 2.0, H);
    // `c.plug_in` is the 32-node rule and `f.plug_in` the 64-node rule: each is closer to
    // the exact integral than the one before, and the recorded 16-vs-32 difference
    // exceeds the 32-node rule's own error (it estimates the coarse rule's error).
    let error_32 = (c.plug_in - exact).abs();
    let error_64 = (f.plug_in - exact).abs();
    assert!(error_64 < error_32, "{error_64} vs {error_32}");
    assert!(error_32 < c.quadrature.estimate_error, "{error_32} vs {:?}", c.quadrature);
    assert!(error_32 > 1e-7, "the kink is visible to the 32-node rule: {error_32}");
    assert!(c.quadrature.max_row_error >= c.quadrature.estimate_error);
    // The truth of the kinked curve at a = 2 is recovered by the fitted hinge curve.
    let truth = 2.0 + 3.0 * kernel_hinge(2.0, H, 2.1) + 0.6;
    assert!((f.estimate - truth).abs() < 0.2, "{} vs {truth}", f.estimate);
    // A polynomial fitted curve needs no tolerance: both rules agree to rounding.
    let smooth = SmoothedDoseOptions { quadrature_tolerance: 1e-9, ..options() };
    let polynomial = antecedent_estimate::estimate_smoothed_dose(
        &id,
        &q,
        &draw(Design::IndependentSamples, Scenario::Good, 3000, 2000, 31),
        &smooth,
        &ExecutionContext::for_tests(31),
    )
    .unwrap();
    assert!(polynomial.grid[0].quadrature.estimate_error < 1e-12);
}

#[test]
fn thread_budget_changes_leave_seeded_results_identical() {
    let q = query(&[1.5, 2.5], H);
    let id = identify(&q);
    for design in DESIGNS {
        let input = draw(design, Scenario::Good, 300, 220, 21);
        let opts = SmoothedDoseOptions { bootstrap: 199, min_local_ess: 10.0, ..options() };
        let serial = ExecutionContext::for_tests(9);
        let mut parallel = ExecutionContext::for_tests(9);
        parallel.parallelism = Parallelism::bounded(NonZeroThreadCount::new(4).unwrap());
        let a = smoothed_dose_interval_internal(&id, &q, &input, &opts, &serial).unwrap();
        let b = smoothed_dose_interval_internal(&id, &q, &input, &opts, &parallel).unwrap();
        assert_eq!(a.replicates, b.replicates);
        assert_eq!(a.failures, b.failures);
        assert_eq!(a.intervals, b.intervals);
        assert_eq!(a.point, b.point);
        assert_eq!(a.replicates.len() + a.failures as usize, 199);
        assert!(a.replicates.windows(2).all(|w| w[0].0 < w[1].0));
        // The public point route is invariant too.
        let point = |ctx: &ExecutionContext| {
            antecedent_estimate::estimate_smoothed_dose(&id, &q, &input, &opts, ctx).unwrap()
        };
        assert_eq!(point(&serial), point(&parallel));
    }
}

#[test]
fn permuting_the_grid_permutes_the_results_bit_for_bit() {
    let forward = [1.0, 1.5, 2.0, 3.0];
    let permuted = [3.0, 1.0, 2.0, 1.5];
    let input = draw(Design::NestedCohort, Scenario::Good, 1500, 1000, 12);
    let run = |grid: &[f64]| {
        let q = query(grid, H);
        antecedent_estimate::estimate_smoothed_dose(
            &identify(&q),
            &q,
            &input,
            &options(),
            &ExecutionContext::for_tests(12),
        )
        .unwrap()
    };
    let a = run(&forward);
    let b = run(&permuted);
    for point in &b.grid {
        let same = a.grid.iter().find(|p| p.dose == point.dose).unwrap();
        assert_eq!(same, point, "grid dose {}", point.dose);
        assert_eq!(same.estimate.to_bits(), point.estimate.to_bits());
    }
    assert_eq!(b.grid.iter().map(|p| p.dose).collect::<Vec<_>>(), permuted);
    assert_eq!(a.models, b.models);
}

#[test]
fn the_bandwidth_is_part_of_the_estimand_identity() {
    let input = draw(Design::IndependentSamples, Scenario::Good, 3000, 2000, 14);
    let run = |h: f64| {
        let builder = prepare(query(&[2.0], h), input.clone(), options(), 14).unwrap();
        builder.estimate(&ExecutionContext::for_tests(14)).unwrap()
    };
    let (narrow, wide) = (run(0.4), run(1.0));
    // A different bandwidth is a different target: closed forms differ by (1 - 0.16)/5.
    let gap = Scenario::Good.truth(2.0, 1.0) - Scenario::Good.truth(2.0, 0.4);
    assert!((gap - 0.168).abs() < 1e-12);
    let moved = wide.estimate().grid[0].estimate - narrow.estimate().grid[0].estimate;
    assert!((moved - gap).abs() < 0.08, "{moved} vs {gap}");
    // The bandwidth is bound into the premises digest and the execution identity.
    assert_ne!(narrow.wire().premises_digest, wide.wire().premises_digest);
    assert_eq!(narrow.wire().data_digest, wide.wire().data_digest);
    assert_ne!(narrow.identity(), wide.identity());
    assert_eq!(narrow.wire().query.bandwidth, 0.4);
}

/// Refit one nuisance on `train` rows alone and return its portable map.
fn refit(
    design: &[f64],
    rows: usize,
    columns: usize,
    target: &[f64],
    train: &[usize],
    spec: LearnerSpec,
    task: antecedent_learn::PredictionTask,
) -> antecedent_learn::PortablePredictor {
    use antecedent_learn::{DesignView, RowSelection, TargetView, resolve_for};
    let ctx = ExecutionContext::for_tests(0);
    let view = DesignView::from_column_major(design, rows, columns).unwrap();
    let train: Vec<u32> = train.iter().map(|i| *i as u32).collect();
    resolve_for(spec, task)
        .unwrap()
        .fit(
            view.with_rows(RowSelection::new(&train)).unwrap(),
            TargetView::new(target),
            None,
            &ctx,
        )
        .unwrap()
        .portable()
        .unwrap()
}

/// Every stored fold model of `result` equals a refit on the rows outside that fold of
/// `input` alone (membership on `[1, z]`, outcome on the default basis of the source rows).
fn assert_models_out_of_fold(
    result: &SmoothedDoseEstimate,
    input: &SmoothedDoseInput,
    opts: &SmoothedDoseOptions,
) {
    use antecedent_learn::PredictionTask::{BinaryProbability, Regression};
    let n = input.source.len();
    let z = &input.covariates[0];
    let mut membership = vec![1.0; n];
    membership.extend_from_slice(z);
    let labels: Vec<f64> = input.source.iter().map(|s| f64::from(*s)).collect();
    let source: Vec<usize> = (0..n).filter(|i| input.source[*i]).collect();
    let m = source.len();
    let mut outcome = vec![1.0; m];
    let dose: Vec<f64> = source.iter().map(|i| input.dose[*i]).collect();
    let x: Vec<f64> = source.iter().map(|i| z[*i]).collect();
    outcome.extend(dose.iter());
    outcome.extend(dose.iter().map(|t| t * t));
    outcome.extend(x.iter());
    outcome.extend(dose.iter().zip(&x).map(|(t, v)| t * v));
    outcome.extend(dose.iter().zip(&x).map(|(t, v)| t * t * v));
    let y: Vec<f64> = source.iter().map(|i| input.outcome[*i]).collect();
    let folds = &result.folds.assignment;
    for k in 0..opts.folds {
        let outside: Vec<usize> = (0..n).filter(|i| usize::from(folds[*i]) != k).collect();
        let reference =
            refit(&membership, n, 2, &labels, &outside, opts.membership, BinaryProbability);
        assert_eq!(reference.model, result.models.membership[k].model, "membership fold {k}");
        let outside: Vec<usize> = (0..m).filter(|r| usize::from(folds[source[*r]]) != k).collect();
        let reference = refit(&outcome, m, 6, &y, &outside, opts.outcome, Regression);
        assert_eq!(reference.model, result.models.outcome[k].model, "outcome fold {k}");
    }
}

/// What this proves: every stored fold model is fitted on the other folds alone, and
/// editing a fold's own outcomes, covariates and doses leaves its models bit-identical.
/// The dose basis is a row-wise transform with no fitted statistic, so preprocessing
/// inside folds holds trivially today; a future full-sample transform would move the
/// models off the reference refit and fail this test.
#[test]
fn every_fold_model_is_fitted_out_of_fold_and_fold_edits_never_reach_it() {
    let q = query(&GRID, H);
    let id = identify(&q);
    let opts = options();
    let input = draw(Design::IndependentSamples, Scenario::Good, 900, 600, 4);
    let ctx = ExecutionContext::for_tests(4);
    let base = antecedent_estimate::estimate_smoothed_dose(&id, &q, &input, &opts, &ctx).unwrap();
    assert_eq!(base.folds.assignment, smoothed_dose_fold_assignment(&input, 3));
    assert_models_out_of_fold(&base, &input, &opts);
    // Edit everything about fold 0's own rows: outcomes, covariates and doses.
    let folds = base.folds.assignment.clone();
    let mut edited = input.clone();
    for i in 0..edited.source.len() {
        if folds[i] == 0 {
            if edited.source[i] {
                edited.outcome[i] += 100.0;
                edited.dose[i] = 4.0 - edited.dose[i];
            }
            edited.covariates[0][i] = 0.3 + 0.9 * edited.covariates[0][i];
        }
    }
    assert_eq!(smoothed_dose_fold_assignment(&edited, 3), folds);
    let moved = antecedent_estimate::estimate_smoothed_dose(&id, &q, &edited, &opts, &ctx).unwrap();
    // Fold 0's models never saw fold 0's rows: they are unchanged bit for bit.
    assert_eq!(moved.models.outcome[0], base.models.outcome[0]);
    assert_eq!(moved.models.membership[0], base.models.membership[0]);
    // The other folds train on the edited rows, so their models do move.
    assert_ne!(moved.models.outcome[1], base.models.outcome[1]);
    assert_ne!(moved.models.membership[2], base.models.membership[2]);
    assert_models_out_of_fold(&moved, &edited, &opts);
    // Perturb membership labels: source rows of fold 0 become target rows. Labels define
    // the fold strata, so the assignment is recomputed; the identity holds under it.
    let mut relabelled = input.clone();
    let flipped: Vec<usize> =
        (0..input.source.len()).filter(|i| folds[*i] == 0 && input.source[*i]).take(20).collect();
    assert_eq!(flipped.len(), 20);
    for &i in &flipped {
        relabelled.source[i] = false;
    }
    let relabelled_folds = smoothed_dose_fold_assignment(&relabelled, 3);
    assert_ne!(relabelled_folds, folds);
    let after =
        antecedent_estimate::estimate_smoothed_dose(&id, &q, &relabelled, &opts, &ctx).unwrap();
    assert_eq!(after.folds.assignment, relabelled_folds);
    assert_models_out_of_fold(&after, &relabelled, &opts);
}

#[test]
fn bootstrap_grouping_follows_the_declared_sampling_design() {
    let ctx = ExecutionContext::for_tests(3);
    let trial_rows = |input: &SmoothedDoseInput, replicate: u32| {
        let groups = smoothed_dose_bootstrap_groups(input);
        let rows =
            antecedent_estimate::learned_trial::bootstrap_rows(&groups, replicate, &ctx).unwrap();
        assert_eq!(rows.len(), input.source.len());
        rows.iter().filter(|i| input.source[**i]).count()
    };
    let independent = draw(Design::IndependentSamples, Scenario::Good, 80, 40, 1);
    assert_eq!(smoothed_dose_bootstrap_groups(&independent).len(), 2);
    assert!((0..40).all(|r| trial_rows(&independent, r) == 80));
    let nested = draw(Design::NestedCohort, Scenario::Good, 80, 40, 1);
    assert_eq!(smoothed_dose_bootstrap_groups(&nested).len(), 1);
    let counts: std::collections::BTreeSet<_> = (0..40).map(|r| trial_rows(&nested, r)).collect();
    assert!(counts.len() > 1, "{counts:?}");
    let groups = smoothed_dose_bootstrap_groups(&independent);
    let draw_of = |r| antecedent_estimate::learned_trial::bootstrap_rows(&groups, r, &ctx).unwrap();
    assert_eq!(draw_of(7), draw_of(7));
    assert_ne!(draw_of(7), draw_of(8));
}

#[test]
fn estimate_reuses_the_prepared_certificate_and_refresh_keeps_it() {
    let prepared = prepare(
        query(&GRID, H),
        draw(Design::IndependentSamples, Scenario::Good, 1500, 1000, 7),
        options(),
        7,
    )
    .unwrap();
    let ctx = ExecutionContext::for_tests(7);
    let first = prepared.estimate(&ctx).unwrap();
    assert_eq!(first.identity(), prepared.estimate(&ctx).unwrap().identity());
    let fresh = draw(Design::IndependentSamples, Scenario::Good, 1500, 1000, 8);
    let refreshed = prepared.refresh(fresh.clone(), &ctx).unwrap();
    assert_eq!(refreshed.identification(), prepared.identification());
    assert_eq!(refreshed.query(), prepared.query());
    assert_ne!(refreshed.estimate(&ctx).unwrap().identity(), first.identity());
    let mut nested = fresh.clone();
    nested.sampling = antecedent_estimate::TrialSampling::NestedCohort;
    assert!(prepared.refresh(nested, &ctx).is_err());
}

#[test]
fn refresh_replaces_rows_and_refuses_a_changed_schema() {
    let builder_rows = draw(Design::NestedCohort, Scenario::Good, 1500, 1000, 15);
    let prepared = prepare(query(&GRID, H), builder_rows, options(), 15).unwrap();
    let ctx = ExecutionContext::for_tests(15);
    let fresh = draw(Design::NestedCohort, Scenario::Good, 1500, 1000, 16);
    let refreshed = prepared.refresh(fresh.clone(), &ctx).unwrap();
    let plan = refreshed.estimate(&ctx).unwrap();
    assert_eq!(plan.wire().input, fresh);
    assert_eq!(
        plan.wire().premises_digest,
        prepared.estimate(&ctx).unwrap().wire().premises_digest
    );
    // A changed schema is re-preparation; a snapshot whose density leaves the floor or
    // whose doses thin out is refused by the same validation as preparation.
    let mut wide = fresh.clone();
    wide.features = vec![0, 1];
    assert!(
        matches!(prepared.refresh(wide, &ctx), Err(IoError::Convert(m)) if m.contains("reprepare_required"))
    );
    let mut floored = fresh;
    let row = floored.source.iter().position(|s| *s).unwrap();
    floored.dose_density[row] = 1e-5;
    let (code, message) = refused_code(prepared.refresh(floored, &ctx).unwrap_err());
    assert_eq!(code, "transport_support_failure");
    assert!(message.starts_with("dose_response.dose_density_floor"));
    // A cancelled refresh refuses before reading the rows.
    let cancelled = ExecutionContext::for_tests(15);
    cancelled.cancellation.cancel();
    let (code, _) = refused_code(
        prepared
            .refresh(draw(Design::NestedCohort, Scenario::Good, 90, 60, 1), &cancelled)
            .unwrap_err(),
    );
    assert_eq!(code, "transport_budget_cancel");
}

fn exported(design: Design) -> (Vec<u8>, SmoothedDoseArtifactWire) {
    let prepared =
        prepare(query(&GRID, H), draw(design, Scenario::Good, 900, 600, 13), options(), 13)
            .unwrap();
    let bytes = prepared.estimate(&ExecutionContext::for_tests(13)).unwrap().export().unwrap();
    let wire = SmoothedDoseArtifactWire::decode(&bytes).unwrap();
    (bytes, wire)
}

fn refused(bytes: &[u8]) -> Refusal {
    match consume_smoothed_dose_artifact(bytes, SmoothedDoseConsumeLimits::default()) {
        Err(IoError::SmoothedDose(kind)) => kind,
        other => {
            panic!("expected a typed refusal, got {:?}", other.map(|r| r.identity().to_owned()))
        }
    }
}

#[test]
fn the_exported_grid_is_replayed_by_an_independent_consumer() {
    for design in DESIGNS {
        let (bytes, wire) = exported(design);
        let consumed =
            consume_smoothed_dose_artifact(&bytes, SmoothedDoseConsumeLimits::default()).unwrap();
        assert_eq!(consumed.estimate(), &wire.result);
        assert_eq!(consumed.identity(), wire.identity().unwrap());
        assert_eq!(wire.certificate.formula, "standardize");
        assert_eq!(wire.result.models.outcome.len(), 3);
        assert_eq!(wire.query.grid, GRID);
        assert_eq!(wire.query.kernel, "epanechnikov");
        assert_eq!(wire.variable_names, ["z", "a", "y"]);
        assert_eq!(wire.result.uncertainty.status, "point_only");
        // Consumption re-exports the same bytes without fitting.
        assert_eq!(consumed.export().unwrap(), bytes);
    }
}

fn mutate(
    wire: &SmoothedDoseArtifactWire,
    edit: impl Fn(&mut SmoothedDoseArtifactWire),
    reseal: bool,
) -> Vec<u8> {
    let mut wire = wire.clone();
    edit(&mut wire);
    if reseal {
        wire.premises_digest = wire.expected_premises_digest().unwrap();
        wire.data_digest = wire.expected_data_digest().unwrap();
        wire.evidence_digest = wire.expected_evidence_digest().unwrap();
    }
    wire.export().unwrap()
}

fn coefficient(
    wire: &mut SmoothedDoseArtifactWire,
    role_outcome: bool,
    fold: usize,
    column: usize,
    delta: f64,
) {
    let models = if role_outcome {
        &mut wire.result.models.outcome
    } else {
        &mut wire.result.models.membership
    };
    let PredictionMap::Linear { coefficients, .. } = &mut models[fold].model else {
        panic!("linear")
    };
    coefficients[column] += delta;
}

#[test]
#[allow(clippy::too_many_lines, reason = "one mutation per assertion group")]
fn a_mutated_artifact_fails_independent_consumption_with_a_typed_error() {
    let (bytes, original) = exported(Design::NestedCohort);
    // Unsealed edits of the result fail the evidence digest.
    assert_eq!(
        refused(&mutate(&original, |w| w.result.grid[0].estimate += 1e-9, false)),
        Refusal::EvidenceMismatch
    );
    assert_eq!(
        refused(&mutate(&original, |w| coefficient(w, true, 0, 1, 1e-9), false)),
        Refusal::EvidenceMismatch
    );
    // Re-sealed semantic mutations fail replay for the right reason.
    assert_eq!(
        refused(&mutate(&original, |w| w.result.grid[1].estimate += 1e-9, true)),
        Refusal::PointMismatch
    );
    assert_eq!(
        refused(&mutate(&original, |w| w.result.grid[1].plug_in += 1e-9, true)),
        Refusal::PointMismatch
    );
    // A fold model edit moves the replayed point: the stored point no longer replays.
    assert_eq!(
        refused(&mutate(&original, |w| coefficient(w, true, 1, 2, 1e-3), true)),
        Refusal::PointMismatch
    );
    assert_eq!(
        refused(&mutate(&original, |w| coefficient(w, false, 2, 1, 1e-3), true)),
        Refusal::PointMismatch
    );
    assert_eq!(
        refused(&mutate(&original, |w| w.result.grid[0].quadrature.estimate_error += 1e-12, true)),
        Refusal::QuadratureMismatch
    );
    assert_eq!(
        refused(&mutate(&original, |w| w.result.grid[0].quadrature.nodes = 32, true)),
        Refusal::QuadratureMismatch
    );
    assert_eq!(
        refused(&mutate(
            &original,
            |w| w.result.grid[2].smoothing_bias.local_quadratic_bias += 1e-9,
            true
        )),
        Refusal::SmoothingBiasMismatch
    );
    assert_eq!(
        refused(&mutate(&original, |w| w.result.grid[0].support.distinct_doses += 1, true)),
        Refusal::SupportMismatch
    );
    assert_eq!(
        refused(&mutate(&original, |w| w.result.grid[0].influence_se_diagnostic *= 1.01, true)),
        Refusal::SupportMismatch
    );
    assert_eq!(
        refused(&mutate(&original, |w| w.result.diagnostics.outcome_rmse += 1e-9, true)),
        Refusal::DiagnosticsMismatch
    );
    assert_eq!(
        refused(&mutate(&original, |w| w.result.overlap.max_odds_weight += 1e-9, true)),
        Refusal::OverlapMismatch
    );
    let swapped = mutate(
        &original,
        |w| {
            let other = w.result.folds.assignment.iter().position(|f| *f != 0).unwrap();
            w.result.folds.assignment.swap(0, other);
        },
        true,
    );
    assert_eq!(refused(&swapped), Refusal::FoldMismatch);
    assert!(matches!(
        refused(&mutate(&original, |w| w.result.uncertainty.reason = "x".into(), true)),
        Refusal::UncertaintyMismatch(_)
    ));
    // Provenance and fold models are checked against the learner specs and design.
    assert_eq!(
        refused(&mutate(&original, |w| w.result.provenance[0].version = "9".into(), true)),
        Refusal::ProvenanceMismatch
    );
    assert_eq!(
        refused(&mutate(
            &original,
            |w| {
                for m in &mut w.result.models.outcome {
                    m.provenance.spec = "ridge".into();
                }
                w.result.provenance = w
                    .result
                    .models
                    .membership
                    .iter()
                    .chain(&w.result.models.outcome)
                    .map(|m| m.provenance.clone())
                    .collect();
            },
            true
        )),
        Refusal::ProvenanceMismatch
    );
    assert_eq!(
        refused(&mutate(&original, |w| w.result.models.outcome[0].columns += 1, true)),
        Refusal::ProvenanceMismatch
    );
    // Premises: grid, bandwidth, kernel, support, provenance, options, seed and names.
    for (label, edit) in [
        (
            "grid",
            Box::new(|w: &mut SmoothedDoseArtifactWire| w.query.grid[0] = 1.1)
                as Box<dyn Fn(&mut SmoothedDoseArtifactWire)>,
        ),
        ("bandwidth", Box::new(|w| w.query.bandwidth = 0.45)),
        ("kernel", Box::new(|w| w.query.kernel = "gaussian".into())),
        ("support", Box::new(|w| w.query.dose_support = (0.0, 5.0))),
        ("density", Box::new(|w| w.query.density_provenance = "estimated".into())),
        ("tolerance", Box::new(|w| w.options.quadrature_tolerance = 1e-3)),
        ("seed", Box::new(|w| w.seed += 1)),
        ("names", Box::new(|w| w.variable_names.swap(0, 2))),
    ] {
        assert_eq!(refused(&mutate(&original, edit, false)), Refusal::PremisesMismatch, "{label}");
    }
    // Re-sealed premise mutations fail replay for the right reason.
    assert_eq!(
        refused(&mutate(&original, |w| w.query.grid[0] = 1.1, true)),
        Refusal::PointMismatch
    );
    assert_eq!(
        refused(&mutate(&original, |w| w.query.bandwidth = 0.45, true)),
        Refusal::PointMismatch
    );
    assert!(matches!(
        refused(&mutate(&original, |w| w.query.kernel = "gaussian".into(), true)),
        Refusal::ReplayRefused { code: "route_not_supported", ref message } if message.starts_with("dose_response.kernel_not_supported")
    ));
    assert!(matches!(
        refused(&mutate(&original, |w| w.query.density_provenance = "estimated".into(), true)),
        Refusal::ReplayRefused { code: "route_not_supported", ref message } if message.starts_with("dose_response.estimated_dose_density")
    ));
    assert!(matches!(
        refused(&mutate(&original, |w| w.query.grid[2] = 3.9, true)),
        Refusal::ReplayRefused { code: "transport_support_failure", ref message } if message.starts_with("dose_response.grid_outside_dose_support")
    ));
    // A polynomial curve integrates identically up to rounding under 32/64 nodes: the
    // replay differs in the recorded node counts, or in the point's last bits.
    assert!(matches!(
        refused(&mutate(&original, |w| w.options.quadrature_nodes = 32, true)),
        Refusal::QuadratureMismatch | Refusal::PointMismatch
    ));
    assert!(matches!(
        refused(&mutate(&original, |w| w.options.min_membership_probability = 0.45, true)),
        Refusal::ReplayRefused { code: "transport_support_failure", ref message } if message.starts_with("dose_response.membership_overlap")
    ));
    assert_eq!(
        refused(&mutate(&original, |w| w.options.folds = 4, true)),
        Refusal::UnsupportedSemantics("result shape")
    );
    assert!(matches!(
        refused(&mutate(&original, |w| w.options.folds = 21, true)),
        Refusal::LimitsExceeded(_)
    ));
    assert!(matches!(
        refused(&mutate(&original, |w| w.bounds.max_grid = 64, true)),
        Refusal::LimitsExceeded(_)
    ));
    assert!(matches!(
        refused(&mutate(&original, |w| w.options.bootstrap = 199, true)),
        Refusal::UncertaintyMismatch(_)
    ));
    assert!(matches!(
        refused(&mutate(&original, |w| w.certificate.rule = "forged".into(), true)),
        Refusal::ProofMismatch(_)
    ));
    // Data: rows, doses and known densities are bound by the data digest; re-sealed edits
    // replay to a different point or a refused density.
    let row = (0..original.input.source.len())
        .find(|i| original.input.source[*i] && (original.input.dose[*i] - 2.0).abs() < 0.2)
        .unwrap();
    assert_eq!(
        refused(&mutate(&original, |w| w.input.outcome[row] += 1.0, false)),
        Refusal::DataIdentityMismatch
    );
    assert_eq!(
        refused(&mutate(&original, |w| w.input.dose_density[row] *= 1.1, false)),
        Refusal::DataIdentityMismatch
    );
    assert_eq!(
        refused(&mutate(&original, |w| w.input.outcome[row] += 1.0, true)),
        Refusal::PointMismatch
    );
    assert_eq!(
        refused(&mutate(&original, |w| w.input.dose_density[row] *= 1.1, true)),
        Refusal::PointMismatch
    );
    assert!(matches!(
        refused(&mutate(&original, |w| w.input.dose_density[row] = 0.0, true)),
        Refusal::ReplayRefused { code: "transport_support_failure", ref message } if message.starts_with("dose_response.dose_density_invalid")
    ));
    // A replay refusal carries its registered reason code through IoError.
    let error = consume_smoothed_dose_artifact(
        &mutate(&original, |w| w.input.dose_density[row] = 0.0, true),
        SmoothedDoseConsumeLimits::default(),
    )
    .unwrap_err();
    assert_eq!(error.reason_code(), Some("transport_support_failure"));
    // Format: features, limits, an interval field and a foreign version.
    assert!(matches!(
        refused(&mutate(&original, |w| w.required_features.push("interval".into()), false)),
        Refusal::UnsupportedSemantics(_)
    ));
    assert!(matches!(
        consume_smoothed_dose_artifact(
            &bytes,
            SmoothedDoseConsumeLimits { max_rows: 10, ..Default::default() }
        ),
        Err(IoError::SmoothedDose(Refusal::LimitsExceeded("row count")))
    ));
    let value: ciborium::Value = ciborium::from_reader(bytes.as_slice()).unwrap();
    let ciborium::Value::Map(mut entries) = value else { panic!("a map") };
    entries.push((ciborium::Value::Text("interval".into()), ciborium::Value::Array(vec![])));
    let mut with_interval = Vec::new();
    ciborium::into_writer(&ciborium::Value::Map(entries), &mut with_interval).unwrap();
    assert!(
        consume_smoothed_dose_artifact(&with_interval, SmoothedDoseConsumeLimits::default())
            .is_err()
    );
    assert!(matches!(
        consume_smoothed_dose_artifact(
            &mutate(&original, |w| w.version = 2, false),
            SmoothedDoseConsumeLimits::default()
        ),
        Err(IoError::UnsupportedVersion { version: 2 })
    ));
    assert_eq!(original.check_variable_names(&original.variable_names.clone()), Ok(()));
    let mut relabelled = original.variable_names.clone();
    relabelled.swap(0, 2);
    assert_eq!(original.check_variable_names(&relabelled), Err(Refusal::NamesMismatch));
    // What replay cannot detect: a fold model replaced together with everything it implies
    // (the point, errors and diagnostics recomputed from it) and re-sealed. Replay proves
    // consistency with the stored models, not that the producer fitted them.
    let mut forged = original.clone();
    coefficient(&mut forged, true, 0, 0, 0.25);
    let q = forged.query.to_query().unwrap();
    let replay = antecedent_estimate::smoothed_dose::evaluate_smoothed_dose_models(
        &q,
        &forged.input,
        &forged.options,
        &forged.result.folds.assignment,
        &forged.result.models,
        &ExecutionContext::for_tests(0),
    )
    .unwrap();
    forged.result.grid = replay.grid;
    forged.result.diagnostics = replay.diagnostics;
    forged.result.overlap = replay.overlap;
    let forged = mutate(&forged, |_| {}, true);
    assert!(consume_smoothed_dose_artifact(&forged, SmoothedDoseConsumeLimits::default()).is_ok());
}

#[test]
fn the_io_wire_consumes_bytes_directly_under_its_own_limits() {
    let (bytes, original) = exported(Design::IndependentSamples);
    let limits = SmoothedDoseConsumeLimits::default();
    let consumed = SmoothedDoseArtifactWire::consume(&bytes, limits).unwrap();
    assert_eq!(consumed.wire.result, original.result);
    assert_eq!(consumed.identification, identify(&query(&GRID, H)));
    let coded = |bytes: &[u8], limits| match SmoothedDoseArtifactWire::consume(bytes, limits) {
        Err(IoError::SmoothedDose(kind)) => kind,
        Err(other) => panic!("expected a typed refusal, got {other}"),
        Ok(_) => panic!("expected a refusal"),
    };
    let rows = original.input.source.len();
    assert_eq!(
        coded(&bytes, SmoothedDoseConsumeLimits { max_rows: rows - 1, ..limits }),
        Refusal::LimitsExceeded("row count")
    );
    assert_eq!(
        coded(&bytes, SmoothedDoseConsumeLimits { max_features: 0, ..limits }),
        Refusal::LimitsExceeded("feature count")
    );
    assert!(
        SmoothedDoseArtifactWire::consume(
            &bytes,
            SmoothedDoseConsumeLimits { max_rows: rows, ..limits }
        )
        .is_ok()
    );
    // The stored bounds must be this build's: a forged, larger bound is a limits refusal
    // before any replay work.
    let forged = mutate(&original, |w| w.bounds.max_rows = 10_000_000, true);
    assert_eq!(
        coded(&forged, limits),
        Refusal::LimitsExceeded("stored bounds are not this build's frozen bounds")
    );
    let foreign = mutate(&original, |w| w.required_features = vec!["other_v9".into()], false);
    assert!(matches!(coded(&foreign, limits), Refusal::UnsupportedSemantics(_)));
    assert!(!matches!(
        SmoothedDoseArtifactWire::consume(&bytes[..bytes.len() / 2], limits),
        Err(IoError::SmoothedDose(_))
    ));
}

#[test]
fn the_unmeasured_interval_route_refuses_with_cell_not_licensed() {
    let rows = draw(Design::IndependentSamples, Scenario::Good, 900, 600, 2);
    let prepared = prepare(query(&GRID, H), rows.clone(), options(), 2).unwrap();
    let ctx = ExecutionContext::for_tests(2);
    let (code, message) = refused_code(prepared.interval(&ctx).unwrap_err());
    assert_eq!(code, "cell_not_licensed");
    assert!(message.starts_with("dose_response.interval_withheld"), "{message}");
    // The point remains available and reports the interval withheld, never attached.
    let requested = SmoothedDoseOptions { bootstrap: 199, ..options() };
    let result = prepare(query(&GRID, H), rows, requested, 2).unwrap().estimate(&ctx).unwrap();
    let uncertainty = &result.estimate().uncertainty;
    assert_eq!(
        (uncertainty.status.as_str(), uncertainty.reason.as_str()),
        ("withheld", "cell_not_licensed")
    );
    assert!(!uncertainty.available());
}

#[test]
fn the_estimator_layer_returns_the_grid_and_refuses_directly() {
    let q = query(&GRID, H);
    let id = identify(&q);
    let input = draw(Design::IndependentSamples, Scenario::Good, 2000, 1500, 21);
    let ctx = ExecutionContext::for_tests(21);
    let direct =
        antecedent_estimate::estimate_smoothed_dose(&id, &q, &input, &options(), &ctx).unwrap();
    assert_eq!(direct.grid.len(), 3);
    assert_eq!(direct.folds.assignment, smoothed_dose_fold_assignment(&input, 3));
    assert_eq!(direct.models.outcome.len(), 3);
    // The seeded estimator layer and the prepared facade agree bit for bit.
    let facade = prepare(q.clone(), input.clone(), options(), 21).unwrap().estimate(&ctx).unwrap();
    assert_eq!(facade.estimate(), &direct);
    // A hard memory limit below the workspace refuses before any fit.
    let mut tight = ExecutionContext::for_tests(21);
    tight.memory.hard_limit_bytes = Some(1 << 16);
    let error = antecedent_estimate::estimate_smoothed_dose(&id, &q, &input, &options(), &tight)
        .unwrap_err();
    assert_eq!(estimation_refusal(&error).0, "transport_budget_cancel");
    // Cancellation before execution refuses too.
    let cancelled = ExecutionContext::for_tests(21);
    cancelled.cancellation.cancel();
    let error =
        antecedent_estimate::estimate_smoothed_dose(&id, &q, &input, &options(), &cancelled)
            .unwrap_err();
    assert_eq!(estimation_refusal(&error).0, "transport_budget_cancel");
    // The layer validates on its own.
    let error = antecedent_estimate::estimate_smoothed_dose(
        &id,
        &query(&[3.9], H),
        &input,
        &options(),
        &ctx,
    )
    .unwrap_err();
    assert!(estimation_refusal(&error).1.starts_with("dose_response.grid_outside_dose_support"));
}

#[test]
fn dose_menu_lists_refused_estimators_with_reasons() {
    let (diagram, _) = diagram();
    let q = query(&GRID, H);
    let menu = StudyBuilder::smoothed_dose_menu(&diagram, &q, None).unwrap();
    let by_name = |m: &antecedent_estimate::EstimatorMenu, name: &str| {
        m.entries.iter().find(|e| e.estimator == name).unwrap().clone()
    };
    assert_eq!(menu.selection, "manual");
    assert!(!serde_json::to_string(&menu).unwrap().contains("recommend"));
    let smoothed = by_name(&menu, "smoothed_dose_transport_aipw");
    assert!(smoothed.eligible && smoothed.static_fields.is_empty());
    assert!(
        smoothed
            .required_laws
            .iter()
            .any(|l| l.contains("X=[v0]") && l.contains("[a - 0.5, a + 0.5]"))
    );
    assert!(smoothed.nuisance_tasks.iter().all(|t| t.contains("default")));
    for (name, detail) in [
        ("kennedy_local_linear_point_curve", Some("dose_response.target_not_smoothed")),
        ("generalized_propensity_estimated_density", Some("dose_response.estimated_dose_density")),
        ("dose_response_simultaneous_band", Some("dose_response.cate_or_simultaneous")),
        ("conditional_group_dose_response", None),
    ] {
        let entry = by_name(&menu, name);
        assert!(!entry.eligible && entry.static_fields.len() == 6, "{name}");
        assert_eq!(entry.refusal.unwrap().detail.as_deref(), detail, "{name}");
    }
    for entry in &menu.entries {
        assert_eq!(entry.eligible, entry.refusal.is_none(), "{}", entry.estimator);
        assert!(!entry.required_laws.is_empty() && !entry.nuisance_tasks.is_empty());
    }
    // Computed fields follow the query and the options in force.
    let tuned =
        SmoothedDoseOptions { folds: 7, quadrature_nodes: 32, min_local_ess: 50.0, ..options() };
    let tuned_menu =
        StudyBuilder::smoothed_dose_menu(&diagram, &query(&[2.0], 0.8), Some(&tuned)).unwrap();
    let entry = by_name(&tuned_menu, "smoothed_dose_transport_aipw");
    assert!(
        entry.nuisance_tasks[0].contains("7-fold") && !entry.nuisance_tasks[0].contains("default")
    );
    assert!(entry.support_requirements.iter().any(|s| s.contains("32-node and 64-node")));
    assert!(entry.support_requirements.iter().any(|s| s.contains("at least 50")));
    assert!(entry.required_laws[0].contains("[a - 0.8, a + 0.8]"));
    // Query and provider refusals.
    let extrapolative =
        StudyBuilder::smoothed_dose_menu(&diagram, &query(&[3.9], H), None).unwrap();
    let entry = by_name(&extrapolative, "smoothed_dose_transport_aipw");
    assert_eq!(
        entry.refusal.unwrap().detail.as_deref(),
        Some("dose_response.grid_outside_dose_support")
    );
    let neural = SmoothedDoseOptions {
        outcome: LearnerSpec::NeuralNet(antecedent_estimate::NeuralSpec::default()),
        ..options()
    };
    let refused_menu = StudyBuilder::smoothed_dose_menu(&diagram, &q, Some(&neural)).unwrap();
    assert_eq!(
        by_name(&refused_menu, "smoothed_dose_transport_aipw").refusal.unwrap().detail.as_deref(),
        Some("dose_response.learner_not_portable")
    );
    // A graph whose selection acts on the outcome certifies nothing.
    let mut graph = antecedent_graph::Admg::with_variables(3);
    for (from, to) in [(0, 2), (1, 2)] {
        graph
            .insert_directed(
                antecedent_graph::DenseNodeId::from_raw(from),
                antecedent_graph::DenseNodeId::from_raw(to),
            )
            .unwrap();
    }
    let blocked = antecedent_graph::SelectionDiagram::try_new(
        graph,
        vec![antecedent_core::VariableId::from_raw(2)],
    )
    .unwrap();
    let menu = StudyBuilder::smoothed_dose_menu(&blocked, &q, None).unwrap();
    assert_eq!(
        by_name(&menu, "smoothed_dose_transport_aipw").refusal.unwrap().code,
        "transport_not_certified"
    );
    // The binary-contrast menu lists the smoothed estimator as refused.
    let binary_query = antecedent_core::TransportQuery::new(
        antecedent_core::ResponseQuery::new(antecedent_core::ResponseFunctional::MeanCurve {
            outcome: antecedent_core::VariableId::from_raw(2),
            treatment: antecedent_core::ContinuousDomain::new(
                antecedent_core::VariableId::from_raw(1),
                antecedent_core::GridSpec::Values(std::sync::Arc::from([0.0, 1.0])),
            ),
        }),
        "trial",
        "target",
        [antecedent_core::VariableId::from_raw(1)],
    );
    let binary = StudyBuilder::learned_continuous_menu(&diagram, &binary_query, None).unwrap();
    let entry = by_name(&binary, "smoothed_dose_transport_aipw");
    assert!(!entry.eligible && entry.refusal.unwrap().code == "route_not_supported");
    // The prepared study exposes its own menu without fitting.
    let prepared =
        prepare(q, draw(Design::NestedCohort, Scenario::Good, 900, 600, 1), options(), 1).unwrap();
    let plan = prepared.estimator_menu();
    assert_eq!(by_name(&plan, "smoothed_dose_transport_aipw").sampling_design, ["nested_cohort"]);
}

#[test]
fn the_free_dose_menu_follows_its_options_without_the_builder() {
    let q = query(&GRID, H);
    let id = identify(&q);
    let defaulted = antecedent_estimate::smoothed_dose_estimator_menu(&id, &q, None, None);
    let explicit = antecedent_estimate::smoothed_dose_estimator_menu(
        &id,
        &q,
        Some(&SmoothedDoseOptions {
            outcome: LearnerSpec::Linear(LinearSpec::default()),
            bootstrap: 300,
            ..options()
        }),
        Some(antecedent_estimate::TrialSampling::IndependentSamples),
    );
    let (a, b) = (&defaulted.entries[0], &explicit.entries[0]);
    assert_eq!(a.estimator, "smoothed_dose_transport_aipw");
    assert!(a.nuisance_tasks.iter().all(|t| t.contains("default: none requested")));
    assert!(b.nuisance_tasks.iter().all(|t| !t.contains("default")));
    assert!(
        b.uncertainty_status.starts_with("withheld: cell_not_licensed")
            && b.uncertainty_status.contains("300")
    );
    assert_eq!(b.sampling_design, ["independent_samples"]);
    assert_eq!(a.sampling_design.len(), 2);
    // Pure: it fits nothing and repeats.
    assert_eq!(
        serde_json::to_string(&defaulted).unwrap(),
        serde_json::to_string(&antecedent_estimate::smoothed_dose_estimator_menu(
            &id, &q, None, None
        ))
        .unwrap()
    );
    let estimated = SmoothedDoseTransportQuery { density_provenance: "estimated".into(), ..q };
    let refused = antecedent_estimate::smoothed_dose_estimator_menu(&id, &estimated, None, None);
    assert_eq!(
        refused.entries[0].refusal.as_ref().unwrap().detail.as_deref(),
        Some("dose_response.estimated_dose_density")
    );
}

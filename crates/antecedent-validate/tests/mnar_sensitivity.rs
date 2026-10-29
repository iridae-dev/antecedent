//! Missing-not-at-random sensitivity (hand-derived oracles).
//!
//! Shift data: treated arm 10 rows with outcomes 1..=6 observed and 4 missing (mean 3.5,
//! missing fraction f1 = 0.4); control arm 10 rows with outcomes 0..=4 observed and 5 missing
//! (mean 2, f0 = 0.5). The MAR effect is 1.5 and the shift model gives
//! `effect(delta) = 1.5 + delta (l1 f1 - l0 f0)`:
//!
//! * treated-only loading: slope 0.4, the sign tips at `delta = -3.75` and the decision
//!   "act when the effect exceeds 1" at `delta = -1.25`;
//! * common loading: slope -0.1, the sign tips at `delta = 15`.
//!
//! The conditional variance is `s^2 / n_obs + (l delta)^2 f (1 - f) / n` per arm
//! (`s^2` 3.5 and 2.5, `n_obs` 6 and 5): at common `delta = 2` it is
//! `3.5/6 + 4 (0.24)/10 + 2.5/5 + 4 (0.25)/10 = 1.279333...`.
//!
//! Tilt data: treated arm 10 rows, 6 observed (3 ones, 3 zeros) and 4 missing; control arm 8
//! rows all observed (2 ones, 6 zeros, mean 0.25). With the tilt on the treated arm the
//! missing-pattern mean is `expit(theta)` (odds `e^theta` times the observed odds 1), so
//! `effect(theta) = 0.3 + 0.4 expit(theta) - 0.25`, the decision "act when the effect exceeds
//! 0.3" tips at `theta = ln(5/3)` and the sign never tips.

#![allow(clippy::float_cmp, reason = "exact grid endpoints and identical computations")]

use antecedent_core::AssumptionStatus;
use antecedent_validate::mnar_sensitivity::{
    mnar_sensitivity, ArmLoading, MnarConclusionKind, MnarData, MnarDecision, MnarEstimator,
    MnarMechanism, MnarModel, MnarSensitivityError, MnarSensitivitySpec,
    MNAR_ASSUMPTION_RANGE_CLAIM, MNAR_MAX_TILT,
};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() <= tolerance, "{actual} vs {expected}");
}

fn shift_data() -> (Vec<f64>, Vec<Option<f64>>) {
    let mut treatment = vec![1.0; 10];
    treatment.extend(vec![0.0; 10]);
    let mut outcome: Vec<Option<f64>> = (1..=6).map(|v| Some(f64::from(v))).collect();
    outcome.extend([None; 4]);
    outcome.extend((0..=4).map(|v| Some(f64::from(v))));
    outcome.extend([None; 5]);
    (treatment, outcome)
}

fn tilt_data() -> (Vec<f64>, Vec<Option<f64>>) {
    let mut treatment = vec![1.0; 10];
    treatment.extend(vec![0.0; 8]);
    let mut outcome: Vec<Option<f64>> = vec![Some(1.0); 3];
    outcome.extend(vec![Some(0.0); 3]);
    outcome.extend([None; 4]);
    outcome.extend(vec![Some(1.0); 2]);
    outcome.extend(vec![Some(0.0); 6]);
    (treatment, outcome)
}

fn shift(loading: ArmLoading, lower: f64, upper: f64) -> MnarSensitivitySpec {
    MnarSensitivitySpec::new(
        MnarMechanism { model: MnarModel::DeltaPatternMixture, loading },
        lower,
        upper,
    )
}

fn run(
    data: &(Vec<f64>, Vec<Option<f64>>),
    spec: &MnarSensitivitySpec,
) -> Result<antecedent_validate::mnar_sensitivity::MnarSensitivityResult, MnarSensitivityError> {
    mnar_sensitivity(&MnarData::new(&data.0, &data.1), spec)
}

#[test]
fn mnar_delta_zero_reproduces_the_mar_estimate() {
    let data = shift_data();
    let result = run(&data, &shift(ArmLoading::common(), -10.0, 10.0)).unwrap();
    close(result.mar.effect, 1.5, 1e-12);
    close(result.mar.effect, result.observed_case_effect, 1e-12);
    close(result.mar.treated_mean, 3.5, 1e-12);
    close(result.mar.control_mean, 2.0, 1e-12);
    // The grid holds the MAR point (21 points over [-10, 10] include zero).
    let at_zero = result.grid.iter().find(|p| p.parameter == 0.0).unwrap();
    close(at_zero.effect, 1.5, 1e-12);
    // The tilt model is also MAR at theta = 0.
    let tilt = tilt_data();
    let spec = MnarSensitivitySpec::new(
        MnarMechanism {
            model: MnarModel::ExponentialTiltingSelection,
            loading: ArmLoading::treated_only(),
        },
        -2.0,
        2.0,
    );
    let result = run(&tilt, &spec).unwrap();
    close(result.mar.effect, result.observed_case_effect, 1e-12);
    close(result.mar.effect, 0.25, 1e-12);
}

#[test]
fn mnar_shift_changes_the_arm_mean_by_delta_times_the_missing_fraction() {
    let data = shift_data();
    let result = run(&data, &shift(ArmLoading::treated_only(), -10.0, 10.0)).unwrap();
    for point in &result.grid {
        close(point.treated_mean, 3.5 + 0.4 * point.parameter, 1e-12);
        close(point.control_mean, 2.0, 1e-12);
        close(point.effect, 1.5 + 0.4 * point.parameter, 1e-12);
    }
    let result = run(&data, &shift(ArmLoading::common(), -10.0, 10.0)).unwrap();
    for point in &result.grid {
        close(point.effect, 1.5 - 0.1 * point.parameter, 1e-12);
    }
}

#[test]
fn mnar_shift_tipping_points_match_the_closed_form() {
    let data = shift_data();
    let mut spec = shift(ArmLoading::treated_only(), -10.0, 10.0);
    spec.decisions = vec![MnarDecision { name: "treat".into(), threshold: 1.0 }];
    let result = run(&data, &spec).unwrap();
    let sign = result.tipping.iter().find(|t| t.kind == MnarConclusionKind::Sign).unwrap();
    assert_eq!(sign.at_mar, 1);
    let negative = sign.negative.unwrap();
    close(negative.parameter, -3.75, 1e-9);
    assert!(negative.bracket_lower > negative.parameter);
    // The effect is exactly zero at the closed-form tipping point, so bisection may land on
    // the boundary itself (sign 0) or just past it (sign -1); never back on the MAR sign.
    assert!(negative.changes_to <= 0);
    assert!(sign.positive.is_none());

    let treat = &result.decisions[0];
    assert!(treat.acts_at_mar && treat.changes_within_range);
    close(treat.tipping.negative.unwrap().parameter, -1.25, 1e-9);
    assert!(treat.tipping.positive.is_none());

    let spec = shift(ArmLoading::common(), -20.0, 20.0);
    let result = run(&data, &spec).unwrap();
    let sign = result.tipping.iter().find(|t| t.kind == MnarConclusionKind::Sign).unwrap();
    close(sign.positive.unwrap().parameter, 15.0, 1e-9);
    assert!(sign.negative.is_none());
}

#[test]
fn mnar_sampling_interval_is_the_delta_method_interval_at_each_point() {
    let data = shift_data();
    let mut spec = shift(ArmLoading::common(), -10.0, 10.0);
    spec.level = 0.95;
    let result = run(&data, &spec).unwrap();
    let point = result.grid.iter().find(|p| p.parameter == 2.0).unwrap();
    let variance: f64 = 3.5 / 6.0 + 4.0 * 0.24 / 10.0 + 2.5 / 5.0 + 4.0 * 0.25 / 10.0;
    close(point.sampling.standard_error, variance.sqrt(), 1e-12);
    let z = 1.959_963_984_540_054;
    close(point.sampling.lower, point.effect - z * variance.sqrt(), 1e-9);
    close(point.sampling.upper, point.effect + z * variance.sqrt(), 1e-9);
    // At MAR the variance has no missingness-rate term.
    close(result.mar.sampling.standard_error, (3.5 / 6.0 + 2.5 / 5.0_f64).sqrt(), 1e-12);
}

#[test]
fn mnar_excludes_null_tips_between_the_grid_points_where_the_interval_flips() {
    let data = shift_data();
    let result = run(&data, &shift(ArmLoading::treated_only(), -10.0, 10.0)).unwrap();
    let null = result.tipping.iter().find(|t| t.kind == MnarConclusionKind::ExcludesNull).unwrap();
    let excludes = |p: &antecedent_validate::mnar_sensitivity::MnarPoint| {
        p.sampling.lower > 0.0 || p.sampling.upper < 0.0
    };
    let mar_excludes = excludes(&result.mar);
    for side in [null.negative, null.positive].into_iter().flatten() {
        let before = result
            .grid
            .iter()
            .filter(|p| {
                (p.parameter.abs() < side.parameter.abs()) && p.parameter * side.parameter >= 0.0
            })
            .all(|p| excludes(p) == mar_excludes);
        assert!(before, "grid points before the tipping point keep the MAR conclusion");
        assert!(side.bracket_lower.abs() < side.parameter.abs());
    }
    assert!(null.negative.is_some() || null.positive.is_some());
    // Closed form: with the treated arm loaded, effect(d) = 1.5 + 0.4 d and
    // Var(d) = 3.5/6 + 0.24/10 d^2 + 2.5/5. The 95% interval first excludes zero where
    // (1.5 + 0.4 d)^2 = z^2 Var(d), the positive root of a quadratic; the other root (about
    // -19.2) lies outside the grid, so there is no negative tipping point.
    let z2 = 1.959_963_984_540_054_f64.powi(2);
    let a = 0.16 - z2 * 0.024;
    let b = 1.2;
    let c = 2.25 - z2 * (3.5 / 6.0 + 0.5);
    let root = (-b + (b * b - 4.0 * a * c).sqrt()) / (2.0 * a);
    assert!(root > 1.0 && root < 2.0, "{root}");
    assert!(null.negative.is_none());
    close(null.positive.unwrap().parameter, root, 1e-6);
}

#[test]
fn mnar_tilt_selection_matches_the_closed_form_and_its_tipping_point() {
    let data = tilt_data();
    let mut spec = MnarSensitivitySpec::new(
        MnarMechanism {
            model: MnarModel::ExponentialTiltingSelection,
            loading: ArmLoading::treated_only(),
        },
        -5.0,
        5.0,
    );
    spec.decisions = vec![MnarDecision { name: "treat".into(), threshold: 0.3 }];
    let result = run(&data, &spec).unwrap();
    for point in &result.grid {
        let expit = 1.0 / (1.0 + (-point.parameter).exp());
        close(point.effect, 0.05 + 0.4 * expit, 1e-12);
    }
    let treat = &result.decisions[0];
    assert!(!treat.acts_at_mar);
    close(treat.tipping.positive.unwrap().parameter, (5.0_f64 / 3.0).ln(), 1e-9);
    assert!(treat.tipping.negative.is_none());
    let sign = result.tipping.iter().find(|t| t.kind == MnarConclusionKind::Sign).unwrap();
    assert!(sign.negative.is_none() && sign.positive.is_none());
    // The tilt is bounded by the observed outcome range: the missing-pattern mean never
    // leaves [0, 1], so the treated mean stays in [0.3, 0.7].
    assert!(result.grid.iter().all(|p| (0.3 - 1e-12..=0.7 + 1e-12).contains(&p.treated_mean)));
}

#[test]
fn mnar_ranges_are_assumption_ranges_never_confidence_intervals() {
    let data = shift_data();
    let result = run(&data, &shift(ArmLoading::treated_only(), -10.0, 10.0)).unwrap();
    assert_eq!(result.inference_claim, MNAR_ASSUMPTION_RANGE_CLAIM);
    assert!(!result.assumption_range.is_confidence_interval);
    assert!(!result.assumption_range.empirically_established);
    assert_eq!(result.empirical_status, "not_empirically_established");
    assert!(result.interpretation.contains("not a confidence interval"));
    assert!(result.sampling_basis.contains("never combined with it"));
    let (low, high) = result.grid.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
        (lo.min(p.effect), hi.max(p.effect))
    });
    close(result.assumption_range.effect_lower, low, 0.0);
    close(result.assumption_range.effect_upper, high, 0.0);
    // The sampling interval at a point is wider than zero and distinct from the range.
    assert!(result.grid.iter().all(|p| p.sampling.standard_error > 0.0));
    let departure = result
        .assumptions
        .iter()
        .find(|a| a.id == "outcome_missingness_departure_from_mar")
        .unwrap();
    assert_eq!(departure.status, AssumptionStatus::Untestable);
}

#[test]
fn mnar_refuses_unsupported_estimators_unbounded_ranges_and_undeclared_mechanisms() {
    let data = shift_data();
    let err = MnarEstimator::from_name("causal_forest").unwrap_err();
    assert_eq!(err.detail(), "mnar_sensitivity.unsupported_estimator");
    assert_eq!(err.reason_code(), "route_not_supported");
    assert_eq!(
        MnarEstimator::from_name("difference_in_means").unwrap().name(),
        "difference_in_means"
    );

    let mut spec = shift(ArmLoading::common(), -1.0, 1.0);
    spec.mechanism = None;
    let err = run(&data, &spec).unwrap_err();
    assert_eq!(err.detail(), "mnar_sensitivity.mechanism_not_declared");
    assert_eq!(err.reason_code(), "required_option_missing");

    for (lower, upper) in
        [(f64::NEG_INFINITY, 1.0), (-1.0, f64::INFINITY), (-1.0, f64::NAN), (-1.0, 1.0e13)]
    {
        let err = run(&data, &shift(ArmLoading::common(), lower, upper)).unwrap_err();
        assert_eq!(err.detail(), "mnar_sensitivity.unbounded_range", "{lower} {upper}");
    }
    let tilt = MnarSensitivitySpec::new(
        MnarMechanism {
            model: MnarModel::ExponentialTiltingSelection,
            loading: ArmLoading::common(),
        },
        -2.0 * MNAR_MAX_TILT,
        1.0,
    );
    assert_eq!(run(&tilt_data(), &tilt).unwrap_err().detail(), "mnar_sensitivity.unbounded_range");

    let err = run(&data, &shift(ArmLoading::common(), 1.0, 2.0)).unwrap_err();
    assert_eq!(err.detail(), "mnar_sensitivity.range_excludes_mar");

    let mut spec = shift(ArmLoading::common(), -1.0, 1.0);
    spec.sampling_composition = Some("add the interval to the range".into());
    let err = run(&data, &spec).unwrap_err();
    assert_eq!(err.detail(), "mnar_sensitivity.composition_not_licensed");
    assert_eq!(err.reason_code(), "cell_not_licensed");

    let zero = ArmLoading { treated: 0.0, control: 0.0 };
    let err = run(&data, &shift(zero, -1.0, 1.0)).unwrap_err();
    assert_eq!(err.detail(), "mnar_sensitivity.invalid_loading");
}

#[test]
fn mnar_refuses_malformed_data_and_options() {
    let (treatment, outcome) = shift_data();
    let spec = shift(ArmLoading::common(), -1.0, 1.0);
    let short = &outcome[..5];
    let err = mnar_sensitivity(&MnarData::new(&treatment, short), &spec).unwrap_err();
    assert_eq!(err.detail(), "mnar_sensitivity.invalid_data");
    let mut bad = treatment.clone();
    bad[0] = 2.0;
    let err = mnar_sensitivity(&MnarData::new(&bad, &outcome), &spec).unwrap_err();
    assert_eq!(err.detail(), "mnar_sensitivity.invalid_data");
    let mut thin = outcome.clone();
    for slot in thin.iter_mut().take(5) {
        *slot = None;
    }
    let err = mnar_sensitivity(&MnarData::new(&treatment, &thin), &spec).unwrap_err();
    assert_eq!(err.detail(), "mnar_sensitivity.insufficient_observed");
    let mut spec = spec;
    spec.level = 1.0;
    assert_eq!(
        run(&(treatment, outcome), &spec).unwrap_err().detail(),
        "mnar_sensitivity.invalid_level"
    );
}

/// The analysis agrees with the retained standard-library reference
/// (`conformance/validate/mnar_sensitivity/reference.py`) on effects, sampling standard errors
/// and tipping points, including the selection model evaluated through its selection mechanism.
#[test]
fn mnar_matches_the_independent_reference() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/validate/mnar_sensitivity/expected.json");
    let document: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let level = document["level"].as_f64().unwrap();
    for case in document["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let treatment: Vec<f64> = case["data"]["treatment"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let outcome: Vec<Option<f64>> =
            case["data"]["outcome"].as_array().unwrap().iter().map(|v| v.as_f64()).collect();
        let model = match case["model"].as_str().unwrap() {
            "delta" => MnarModel::DeltaPatternMixture,
            _ => MnarModel::ExponentialTiltingSelection,
        };
        let loading = case["loading"].as_array().unwrap();
        let mechanism = MnarMechanism {
            model,
            loading: ArmLoading {
                treated: loading[0].as_f64().unwrap(),
                control: loading[1].as_f64().unwrap(),
            },
        };
        let mut spec = MnarSensitivitySpec::new(
            mechanism,
            case["lower"].as_f64().unwrap(),
            case["upper"].as_f64().unwrap(),
        );
        spec.grid_points = case["grid_points"].as_u64().unwrap() as usize;
        spec.level = level;
        spec.decisions = case["decisions"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, threshold)| MnarDecision {
                name: name.clone(),
                threshold: threshold.as_f64().unwrap(),
            })
            .collect();
        let result = run(&(treatment, outcome), &spec).unwrap();
        for probe in case["probe"].as_array().unwrap() {
            let parameter = probe["parameter"].as_f64().unwrap();
            let point = result
                .grid
                .iter()
                .find(|p| (p.parameter - parameter).abs() < 1e-9)
                .unwrap_or_else(|| panic!("{name}: no grid point at {parameter}"));
            close(point.effect, probe["effect"].as_f64().unwrap(), 1e-9);
            close(point.sampling.standard_error, probe["standard_error"].as_f64().unwrap(), 1e-6);
        }
        for tipping in &result.tipping {
            let expected = &case["tipping"][tipping.label.as_str()];
            for (side, key) in [(tipping.negative, "negative"), (tipping.positive, "positive")] {
                match (side, expected[key].as_f64()) {
                    (None, None) => {}
                    (Some(side), Some(reference)) => close(side.parameter, reference, 1e-7),
                    (side, reference) => {
                        panic!("{name} {} {key}: {side:?} vs {reference:?}", tipping.label)
                    }
                }
            }
        }
    }
}

/// Adjusted data: a binary covariate x with arms mixed in each cell.
///
/// | cell | arm | rows | observed mean | missing |
/// |------|-----|------|---------------|---------|
/// | x=0  | 1   | 12   | 2 (9 rows)    | 3       |
/// | x=1  | 1   | 8    | 5 (6 rows)    | 2       |
/// | x=0  | 0   | 8    | 1 (6 rows)    | 2       |
/// | x=1  | 0   | 12   | 3 (9 rows)    | 3       |
///
/// The covariate models are saturated, so every adjusted estimator equals the stratified
/// contrast `sum_x P(x) (ybar*_{1,x} - ybar*_{0,x})` over the completed cells, with
/// `ybar*_{a,x} = ybar_{a,x} + s_a * missing_{a,x} / rows_{a,x}` and `P(x) = 1/2`:
/// `effect = 1.5 + 0.25 (s_1 - s_0)`. The treated-only shift gives `1.5 + 0.25 delta`.
fn adjusted_data() -> (Vec<f64>, Vec<Option<f64>>, Vec<f64>) {
    let (mut treatment, mut outcome, mut x) = (Vec::new(), Vec::new(), Vec::new());
    let cells: [(f64, f64, [f64; 3], usize, usize); 4] = [
        (1.0, 0.0, [1.0, 2.0, 3.0], 3, 3),
        (1.0, 1.0, [4.0, 5.0, 6.0], 2, 2),
        (0.0, 0.0, [0.0, 1.0, 2.0], 2, 2),
        (0.0, 1.0, [2.0, 3.0, 4.0], 3, 3),
    ];
    for (arm, cell, values, missing, repeats) in cells {
        for _ in 0..repeats {
            for v in values {
                treatment.push(arm);
                outcome.push(Some(v));
                x.push(cell);
            }
        }
        for _ in 0..missing {
            treatment.push(arm);
            outcome.push(None);
            x.push(cell);
        }
    }
    (treatment, outcome, x)
}

fn adjusted_spec(estimator: MnarEstimator, model: MnarModel) -> MnarSensitivitySpec {
    let mut spec = MnarSensitivitySpec::new(
        MnarMechanism { model, loading: ArmLoading::treated_only() },
        -10.0,
        10.0,
    );
    spec.estimator = estimator;
    spec.bootstrap_replicates = 60;
    spec.seed = 7;
    spec
}

#[test]
fn mnar_adjusted_estimators_match_the_stratified_closed_form() {
    let (treatment, outcome, x) = adjusted_data();
    // The treated arm of the first two cells holds 9 + 3 and 6 + 2 rows: 12 and 8.
    assert_eq!(treatment.iter().filter(|t| **t == 1.0).count(), 20);
    let covariates: [&[f64]; 1] = [&x];
    for estimator in [
        MnarEstimator::RegressionAdjusted,
        MnarEstimator::InverseProbabilityWeighting,
        MnarEstimator::AugmentedInverseProbabilityWeighting,
    ] {
        let spec = adjusted_spec(estimator, MnarModel::DeltaPatternMixture);
        let data = MnarData::new(&treatment, &outcome).with_covariates(&covariates);
        let mut spec = spec;
        spec.decisions = vec![MnarDecision { name: "treat".into(), threshold: 1.0 }];
        let result = mnar_sensitivity(&data, &spec).unwrap();
        close(result.mar.effect, 1.5, 1e-9);
        for point in &result.grid {
            close(point.effect, 1.5 + 0.25 * point.parameter, 1e-9);
            assert!(
                point.sampling.standard_error.is_finite() && point.sampling.standard_error > 0.0
            );
        }
        let sign = result.tipping.iter().find(|t| t.kind == MnarConclusionKind::Sign).unwrap();
        close(sign.negative.unwrap().parameter, -6.0, 1e-7);
        close(result.decisions[0].tipping.negative.unwrap().parameter, -2.0, 1e-7);
        // The bootstrap stream is fixed by the seed.
        let again = mnar_sensitivity(&data, &spec).unwrap();
        assert_eq!(again.grid[3].sampling, result.grid[3].sampling);
    }
}

#[test]
fn mnar_adjusted_tilt_is_missing_at_random_at_zero_and_monotone() {
    let (treatment, outcome, x) = adjusted_data();
    let covariates: [&[f64]; 1] = [&x];
    let data = MnarData::new(&treatment, &outcome).with_covariates(&covariates);
    let spec =
        adjusted_spec(MnarEstimator::RegressionAdjusted, MnarModel::ExponentialTiltingSelection);
    let mut spec = spec;
    spec.lower = -1.0;
    spec.upper = 1.0;
    let result = mnar_sensitivity(&data, &spec).unwrap();
    close(result.mar.effect, 1.5, 1e-9);
    assert!(result.grid.windows(2).all(|w| w[1].effect >= w[0].effect - 1e-12));
}

#[test]
fn mnar_adjusted_estimators_need_declared_covariates() {
    let (treatment, outcome, x) = adjusted_data();
    let spec = adjusted_spec(
        MnarEstimator::AugmentedInverseProbabilityWeighting,
        MnarModel::DeltaPatternMixture,
    );
    let err = mnar_sensitivity(&MnarData::new(&treatment, &outcome), &spec).unwrap_err();
    assert_eq!(err.detail(), "mnar_sensitivity.covariates_required");
    let covariates: [&[f64]; 1] = [&x];
    let dim = shift(ArmLoading::common(), -1.0, 1.0);
    let err =
        mnar_sensitivity(&MnarData::new(&treatment, &outcome).with_covariates(&covariates), &dim)
            .unwrap_err();
    assert_eq!(err.detail(), "mnar_sensitivity.covariates_not_used");
}

#[test]
fn mnar_tipping_sees_a_change_that_reverses_inside_one_grid_cell() {
    // Treated-only tilt on binary outcomes with a decision threshold the effect crosses and
    // recrosses is not available in closed form; the cell scan is checked on the shift model
    // with a two-point grid, where the whole range is one cell.
    let data = shift_data();
    let mut spec = shift(ArmLoading::treated_only(), -10.0, 10.0);
    spec.grid_points = 2;
    let result = run(&data, &spec).unwrap();
    let sign = result.tipping.iter().find(|t| t.kind == MnarConclusionKind::Sign).unwrap();
    close(sign.negative.unwrap().parameter, -3.75, 1e-9);
}

#[test]
fn mnar_result_replays_bit_identically_with_its_assumption_records() {
    let (treatment, outcome, x) = adjusted_data();
    let covariates: [&[f64]; 1] = [&x];
    let data = MnarData::new(&treatment, &outcome).with_covariates(&covariates);
    let mut spec = adjusted_spec(
        MnarEstimator::AugmentedInverseProbabilityWeighting,
        MnarModel::DeltaPatternMixture,
    );
    spec.decisions = vec![MnarDecision { name: "treat".into(), threshold: 1.0 }];
    let first = mnar_sensitivity(&data, &spec).unwrap();
    let second = mnar_sensitivity(&data, &spec).unwrap();
    assert_eq!(first, second);
    assert!(first.assumptions.iter().any(|a| a.id == "outcome_missingness_departure_from_mar"
        && a.status == AssumptionStatus::Untestable));
    assert!(first
        .assumptions
        .iter()
        .any(|a| a.id == "treatment_independent_of_potential_outcomes"
            && a.status == AssumptionStatus::Declared));
}

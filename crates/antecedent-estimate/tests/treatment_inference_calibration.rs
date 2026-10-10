//! Frozen full-fit B4 diagnostics, no invented confidence intervals. Vector design
//! uses the full ±1 product(U,V,Z), T1=U+Z,T2=.5U+V+.5Z; closed treatment covariance
//! is(1/n)[[1.25,-.5],[-.5,1]]. Categorical design has three balanced groups and
//! independent known Gaussian errors. `ModelBased` is measured only homoskedastically;
//! HC0/1/2/3 also use declared heteroskedastic variance. Full/partial coefficient nulls
//! test actual Wald/Holm; flat ordered nulls and strong violations test monotonicity.
//! R>=2000, all failures fail, covariance scaled error<=.15, mean error<=5MCSE,
//! nominal Wald .05±3MCSE, Holm/ordered false rejection<=.05+3MCSE, strong power>=.9.
//! All measurements are ignored; public point/asymptotic standing stays unchanged.
#![allow(clippy::cast_precision_loss, reason = "bounded simulation dimensions")]
#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "common/candidate_calibration.rs"]
mod candidate;
#[path = "common/inference_metrics.rs"]
mod metrics;
use antecedent_estimate::categorical_treatment::{
    CategoricalTreatmentInput, CategoricalTreatmentSpec, LevelScale, MonotonicityDirection,
    fit_categorical_treatment,
};
use antecedent_estimate::vector_treatment::{
    Contrast, NamedColumn, TreatmentColumn, VectorCovariance, VectorTreatmentInput,
    VectorTreatmentOptions, fit_vector_treatment,
};
use calibration::{grid_n, grid_seed, n_sim};
use metrics::{FitMetrics, MetricDesign};
fn kinds(hetero: bool) -> Vec<VectorCovariance> {
    let mut v = vec![
        VectorCovariance::Hc0,
        VectorCovariance::Hc1,
        VectorCovariance::Hc2,
        VectorCovariance::Hc3,
    ];
    if !hetero {
        v.insert(0, VectorCovariance::ModelBased);
    }
    v
}
fn vector_input(n: usize, seed: u64, hetero: bool, beta: [f64; 2]) -> VectorTreatmentInput {
    let mut rng = candidate::Generator::new(seed);
    let (mut y, mut t1, mut t2, mut z) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for i in 0..n {
        let sign = |bit: usize| if bit == 0 { -1. } else { 1. };
        let (u, v, w) = (sign(i & 1), sign((i >> 1) & 1), sign((i >> 2) & 1));
        let (a, b) = (u + w, 0.5 * u + v + 0.5 * w);
        let variance = if hetero { 1. + 0.5 * u } else { 1. };
        y.push(1. + 0.3 * w + beta[0] * a + beta[1] * b + variance.sqrt() * rng.normal());
        t1.push(a);
        t2.push(b);
        z.push(w);
    }
    let treatment = |name: &str, values| TreatmentColumn {
        name: name.into(),
        values,
        adjustment_set: vec!["z".into()],
        row_snapshot: "vector_balanced".into(),
    };
    VectorTreatmentInput {
        outcome: y,
        row_snapshot: "vector_balanced".into(),
        adjustment: vec![NamedColumn { name: "z".into(), values: z }],
        treatments: vec![treatment("t1", t1), treatment("t2", t2)],
    }
}
fn vector(test: &str, hetero: bool) {
    let n = grid_n(512);
    let attempts = usize::try_from(n_sim().max(2000)).unwrap();
    let seed = grid_seed(0x48ab_0000);
    for covariance in kinds(hetero) {
        for partial in [false, true] {
            let beta = if partial { [0., 0.5] } else { [0., 0.] };
            let options = VectorTreatmentOptions {
                covariance,
                contrasts: vec![
                    Contrast { name: "t1".into(), weights: vec![("t1".into(), 1.)] },
                    Contrast { name: "t2".into(), weights: vec![("t2".into(), 1.)] },
                    Contrast {
                        name: "sum".into(),
                        weights: vec![("t1".into(), 1.), ("t2".into(), 1.)],
                    },
                ],
            };
            let results: Vec<_> = (0..attempts)
                .map(|rep| {
                    fit_vector_treatment(
                        &vector_input(n, seed + rep as u64, hetero, beta),
                        &options,
                    )
                    .ok()
                    .map(|fit| FitMetrics {
                        rows: fit.n_rows,
                        estimates: [fit.coefficients[0].estimate, fit.coefficients[1].estimate],
                        covariance: fit.covariance.try_into().unwrap(),
                        joint_p: fit.joint_wald.p_value,
                        contrast_p_holm: std::array::from_fn(|i| fit.contrasts[i].p_holm),
                        contrast_variance: std::array::from_fn(|i| {
                            fit.contrasts[i].standard_error.powi(2)
                        }),
                    })
                })
                .collect();
            metrics::assess(
                MetricDesign {
                    test,
                    file: file!(),
                    method: &format!(
                        "vector.{}.hetero={hetero}.partial_null={partial}",
                        covariance.as_str()
                    ),
                    n,
                    seed,
                    true_coefficients: beta,
                    true_covariance: [
                        1.25 / n as f64,
                        -0.5 / n as f64,
                        -0.5 / n as f64,
                        1. / n as f64,
                    ],
                    true_contrasts: [beta[0], beta[1], beta[0] + beta[1]],
                    true_contrast_variance: [1.25 / n as f64, 1. / n as f64, 1.25 / n as f64],
                },
                attempts,
                &results,
            );
        }
    }
}
fn categorical_input(
    n: usize,
    seed: u64,
    hetero: bool,
    means: [f64; 3],
) -> CategoricalTreatmentInput {
    let mut rng = candidate::Generator::new(seed);
    let (mut levels, mut outcome) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let g = i % 3;
        let sd = if hetero { [1., 1.5, 2.][g] } else { 1. };
        levels.push(["a", "b", "c"][g].into());
        outcome.push(means[g] + sd * rng.normal());
    }
    CategoricalTreatmentInput {
        outcome,
        row_snapshot: "categorical_balanced".into(),
        adjustment: vec![],
        levels,
    }
}
fn spec(
    covariance: VectorCovariance,
    direction: Option<MonotonicityDirection>,
) -> CategoricalTreatmentSpec {
    CategoricalTreatmentSpec {
        declared_levels: vec!["a".into(), "b".into(), "c".into()],
        scale: LevelScale::Ordered,
        reference: "a".into(),
        min_level_rows: 2,
        pairwise: vec![("b".into(), "c".into())],
        monotonicity: direction,
        covariance,
    }
}
fn categorical(test: &str, hetero: bool) {
    let n = grid_n(384);
    let attempts = usize::try_from(n_sim().max(2000)).unwrap();
    let seed = grid_seed(0x49ab_0000);
    for covariance in kinds(hetero) {
        for partial in [false, true] {
            let means = if partial { [0., 0., 1.5] } else { [0., 0., 0.] };
            let s = spec(covariance, None);
            let results: Vec<_> = (0..attempts)
                .map(|rep| {
                    fit_categorical_treatment(
                        &categorical_input(n, seed + rep as u64, hetero, means),
                        &s,
                    )
                    .ok()
                    .map(|fit| FitMetrics {
                        rows: fit.counts.iter().map(|c| c.rows).sum(),
                        estimates: [fit.coefficients[0].estimate, fit.coefficients[1].estimate],
                        covariance: fit.covariance.try_into().unwrap(),
                        joint_p: fit.omnibus.p_value,
                        contrast_p_holm: [
                            fit.level_contrasts[0].p_holm,
                            fit.level_contrasts[1].p_holm,
                            fit.pairwise[0].p_holm,
                        ],
                        contrast_variance: [
                            fit.level_contrasts[0].standard_error.powi(2),
                            fit.level_contrasts[1].standard_error.powi(2),
                            fit.pairwise[0].standard_error.powi(2),
                        ],
                    })
                })
                .collect();
            let var = if hetero { [1., 2.25, 4.] } else { [1., 1., 1.] };
            let scale = 3. / n as f64;
            metrics::assess(
                MetricDesign {
                    test,
                    file: file!(),
                    method: &format!(
                        "categorical.{}.hetero={hetero}.partial_null={partial}",
                        covariance.as_str()
                    ),
                    n,
                    seed,
                    true_coefficients: [means[1] - means[0], means[2] - means[0]],
                    true_covariance: [
                        scale * (var[0] + var[1]),
                        scale * var[0],
                        scale * var[0],
                        scale * (var[0] + var[2]),
                    ],
                    true_contrasts: [means[1] - means[0], means[2] - means[0], means[2] - means[1]],
                    true_contrast_variance: [
                        scale * (var[0] + var[1]),
                        scale * (var[0] + var[2]),
                        scale * (var[1] + var[2]),
                    ],
                },
                attempts,
                &results,
            );
        }
    }
}
fn monotonicity(test: &str, hetero: bool) {
    let n = grid_n(384);
    let attempts = usize::try_from(n_sim().max(2000)).unwrap();
    let seed = grid_seed(0x4aab_0000);
    for covariance in kinds(hetero) {
        for direction in
            [MonotonicityDirection::NonDecreasing, MonotonicityDirection::NonIncreasing]
        {
            for alternative in [false, true] {
                let sign = if direction == MonotonicityDirection::NonDecreasing { -1. } else { 1. };
                let means = if alternative { [0., sign * 1.5, sign * 3.] } else { [0., 0., 0.] };
                let s = spec(covariance, Some(direction));
                let results: Vec<_> = (0..attempts)
                    .map(|rep| {
                        fit_categorical_treatment(
                            &categorical_input(n, seed + rep as u64, hetero, means),
                            &s,
                        )
                        .ok()
                        .and_then(|fit| fit.monotonicity.map(|m| m.p_value))
                    })
                    .collect::<Vec<_>>();
                metrics::assess_monotonicity(
                    test,
                    &format!("{}.hetero={hetero}.{direction:?}", covariance.as_str()),
                    n,
                    seed,
                    alternative,
                    &results,
                );
            }
        }
    }
}
#[test]
#[ignore = "calibration: final measurement only"]
fn vector_full_fit_homoskedastic() {
    vector("vector_full_fit_homoskedastic", false);
}
#[test]
#[ignore = "calibration: final measurement only"]
fn vector_full_fit_heteroskedastic() {
    vector("vector_full_fit_heteroskedastic", true);
}
#[test]
#[ignore = "calibration: final measurement only"]
fn categorical_full_fit_homoskedastic() {
    categorical("categorical_full_fit_homoskedastic", false);
}
#[test]
#[ignore = "calibration: final measurement only"]
fn categorical_full_fit_heteroskedastic() {
    categorical("categorical_full_fit_heteroskedastic", true);
}
#[test]
#[ignore = "calibration: final measurement only"]
fn categorical_monotonicity_homoskedastic() {
    monotonicity("categorical_monotonicity_homoskedastic", false);
}
#[test]
#[ignore = "calibration: final measurement only"]
fn categorical_monotonicity_heteroskedastic() {
    monotonicity("categorical_monotonicity_heteroskedastic", true);
}

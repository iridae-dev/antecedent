//! Full original nonlinear mediation and latent-mixture bootstrap SE diagnostics.
//! Mediation: randomized A, X uniform{-1,0,1,2,3}, M=0.5A+0.5X+0.8Z,
//! Y=1+0.8A+0.6M+0.4M²+0.3AM+0.7X+0.5E, independent standard normal Z,E.
//! Exact natural direct/indirect/total effects are [0.95,0.75,1.7].
//! Mixture: iid class probability(0.6,0.4), independently randomized binary A
//! and X uniform{-1,0,1}, normal noiseSD.5; class means A+0.3X and
//! 20-2A+0.3X. Six (A,X) design points support d=3,K=2 under the existing
//! d+K-1 distinct-design guard. Canonical effects[-2,1], weights[0.4,0.6], mean-0.2.
//! Fits use original quadrature+whole-row bootstrap and EM+aligned bootstrap;
//! class labels are never supplied. n250/500/1000 mediation and200/400/800 mixture,
//! R>=1000,B300. Every outer/inner failure fails. Reported bootstrap variances
//! must match independent full-fit empirical variance within20%, bias<=5MCSE.
//! No intervals or general latent-class identification guarantees are created.
#![allow(clippy::cast_precision_loss, reason = "bounded simulation dimensions")]
#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "common/candidate_calibration.rs"]
mod candidate;
#[path = "common/bootstrap_precision.rs"]
mod metrics;
use antecedent_estimate::latent_class_effects::{
    LatentClassConfig, LatentClassData, fit_latent_class_effects,
};
use antecedent_estimate::nonlinear_mediation::{
    NonlinearMediationConfig, NonlinearMediationInput, NonlinearMediationPremises,
    estimate_nonlinear_mediation,
};
use calibration::{grid_n, grid_seed, n_sim};
use metrics::Fit;
fn mediation(n: usize, seed: u64) -> Option<Fit> {
    let mut rng = candidate::Generator::new(seed);
    let (mut a, mut x, mut m, mut y) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n {
        let av = rng.binary(0.5) as f64;
        let xv = (rng.uniform() * 5.).floor() - 1.;
        let mv = 0.5 * av + 0.5 * xv + 0.8 * rng.normal();
        let yv = 1.
            + 0.8 * av
            + 0.6 * mv
            + 0.4 * mv * mv
            + 0.3 * av * mv
            + 0.7 * xv
            + 0.5 * rng.normal();
        a.push(av);
        x.push(xv);
        m.push(mv);
        y.push(yv);
    }
    let input =
        NonlinearMediationInput { treatment: &a, mediator: &m, outcome: &y, covariates: &[&x] };
    let config = NonlinearMediationConfig { seed, bootstrap_replicates: 300, ..Default::default() };
    let f = estimate_nonlinear_mediation(
        &input,
        &NonlinearMediationPremises::sequentially_ignorable(),
        &config,
    )
    .ok()?;
    Some(Fit {
        estimates: vec![f.natural_direct, f.natural_indirect, f.total],
        standard_errors: vec![f.natural_direct_se?, f.natural_indirect_se?, f.total_se?],
        inner_requested: f.bootstrap.replicates_requested as usize,
        inner_succeeded: f.bootstrap.replicates_succeeded as usize,
    })
}
struct MixtureFixture {
    treatment: Vec<f64>,
    outcome: Vec<f64>,
    covariates: [Vec<f64>; 1],
    // Only the independent nonignored oracle reads labels; no estimator receives them.
    generating_class: Vec<usize>,
}
impl MixtureFixture {
    fn data(&self) -> LatentClassData<'_> {
        LatentClassData {
            treatment: &self.treatment,
            outcome: &self.outcome,
            covariates: &self.covariates,
        }
    }
}
fn mixture_fixture(n: usize, seed: u64) -> MixtureFixture {
    let mut rng = candidate::Generator::new(seed);
    let (mut a, mut x, mut y, mut classes) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n {
        let av = rng.binary(0.5) as f64;
        let xv = (rng.uniform() * 3.).floor() - 1.;
        let second = rng.binary(0.4) == 1;
        a.push(av);
        x.push(xv);
        y.push((if second { 20. - 2. * av } else { av }) + 0.3 * xv + 0.5 * rng.normal());
        classes.push(usize::from(second));
    }
    MixtureFixture { treatment: a, outcome: y, covariates: [x], generating_class: classes }
}
fn mixture_config(seed: u64, bootstrap_replicates: usize) -> LatentClassConfig {
    let mut config = LatentClassConfig::new(2, seed).declare_conditional_randomization();
    config.restarts = 10;
    config.max_iterations = 500;
    config.tolerance = 1e-10;
    config.min_class_weight = 0.05;
    config.min_separation = 0.7;
    config.variance_floor = 1e-8;
    config.bootstrap_replicates = bootstrap_replicates;
    config
}
fn mixture(n: usize, seed: u64) -> Option<Fit> {
    let fixture = mixture_fixture(n, seed);
    let data = fixture.data();
    let config = mixture_config(seed, 300);
    let f = fit_latent_class_effects(&data, &config).ok()?;
    let b = f.bootstrap?;
    Some(Fit {
        estimates: vec![
            f.classes[0].effect,
            f.classes[1].effect,
            f.classes[0].weight,
            f.classes[1].weight,
            f.mixture_average_effect,
        ],
        standard_errors: vec![
            f.classes[0].effect_se?,
            f.classes[1].effect_se?,
            f.classes[0].weight_se?,
            f.classes[1].weight_se?,
            f.mixture_average_se?,
        ],
        inner_requested: b.requested,
        inner_succeeded: b.succeeded,
    })
}
#[test]
#[ignore = "calibration final only"]
fn nonlinear_mediation_whole_method_bootstrap_se() {
    let n = grid_n(500);
    let seed = grid_seed(0xB44D_0001);
    let results: Vec<_> =
        (0..n_sim().max(1000)).map(|r| mediation(n, seed.wrapping_add(u64::from(r)))).collect();
    metrics::assess(
        "nonlinear_mediation_whole_method_bootstrap_se",
        n,
        seed,
        &[0.95, 0.75, 1.7],
        &results,
    );
}
#[test]
#[ignore = "calibration final only"]
fn latent_class_whole_method_bootstrap_se() {
    let n = grid_n(400);
    let seed = grid_seed(0xB44C_0001);
    let results: Vec<_> =
        (0..n_sim().max(1000)).map(|r| mixture(n, seed.wrapping_add(u64::from(r)))).collect();
    metrics::assess(
        "latent_class_whole_method_bootstrap_se",
        n,
        seed,
        &[-2., 1., 0.4, 0.6, -0.2],
        &results,
    );
}

/// Three original point fits check supported design and an independent sufficient-
/// statistics oracle before measurement. No repeated calibration or bootstrap runs.
#[test]
fn latent_mixture_fixture_supported_design_and_independent_truth_prerequisite() {
    for n in [200, 400, 800] {
        let fixture = mixture_fixture(n, 0xB44C_0001);
        let points: std::collections::BTreeSet<_> = fixture
            .treatment
            .iter()
            .zip(&fixture.covariates[0])
            .map(|(a, x)| (a.to_bits(), x.to_bits()))
            .collect();
        assert_eq!(points.len(), 6, "all declared A/X points are represented");
        let config = mixture_config(0xB44C_0001, 0); // Prerequisite only, not SE measurement.
        assert!(config.initial_labels.is_none());
        let result = fit_latent_class_effects(&fixture.data(), &config)
            .expect("supported fixture must reach original point fitting");
        assert!(result.bootstrap.is_none());
        let mut oracle = Vec::new();
        for class in 0..2 {
            let rows: Vec<_> = fixture
                .generating_class
                .iter()
                .enumerate()
                .filter_map(|(i, c)| (*c == class).then_some(i))
                .collect();
            let count = rows.len() as f64;
            let a = rows.iter().map(|i| fixture.treatment[*i]).sum::<f64>() / count;
            let x = rows.iter().map(|i| fixture.covariates[0][*i]).sum::<f64>() / count;
            let y = rows.iter().map(|i| fixture.outcome[*i]).sum::<f64>() / count;
            let (mut aa, mut xx, mut ax, mut ay, mut xy) = (0., 0., 0., 0., 0.);
            for i in &rows {
                let da = fixture.treatment[*i] - a;
                let dx = fixture.covariates[0][*i] - x;
                let dy = fixture.outcome[*i] - y;
                aa += da * da;
                xx += dx * dx;
                ax += da * dx;
                ay += da * dy;
                xy += dx * dy;
            }
            let determinant = aa * xx - ax * ax;
            assert!(determinant > 0.);
            let effect = (ay * xx - xy * ax) / determinant;
            let covariate = (xy * aa - ay * ax) / determinant;
            oracle.push((
                effect,
                rows.len() as f64 / n as f64,
                y - effect * a - covariate * x,
                covariate,
            ));
        }
        oracle.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (actual, expected) in result.classes.iter().zip(&oracle) {
            assert!((actual.effect - expected.0).abs() < 1e-8);
            assert!((actual.weight - expected.1).abs() < 1e-8);
            assert!((actual.intercept - expected.2).abs() < 1e-8);
            assert!((actual.covariate_coefficients[0] - expected.3).abs() < 1e-8);
        }
        let average: f64 = oracle.iter().map(|r| r.0 * r.1).sum();
        assert!((result.mixture_average_effect - average).abs() < 1e-8);
        // Removing X must retain the original support refusal, not relax the guard.
        let unsupported = LatentClassData {
            treatment: &fixture.treatment,
            outcome: &fixture.outcome,
            covariates: &[],
        };
        let error = fit_latent_class_effects(&unsupported, &config).unwrap_err();
        assert!(error.to_string().contains("latent_class.design_support_too_small"));
    }
    // Independent finite population contrasts (noise mean zero) at each X point.
    let mut effects = [0.; 2];
    for (class, weight) in [(0, 0.6), (1, 0.4)] {
        for x in [-1., 0., 1.] {
            let mean = |a: f64| (if class == 0 { a } else { 20. - 2. * a }) + 0.3 * x;
            effects[class] += (mean(1.) - mean(0.)) / 3.;
        }
        assert!((effects[class] - (if class == 0 { 1. } else { -2. })).abs() < 1e-14);
        assert!(weight > 0.);
    }
    assert!((0.6 * effects[0] + 0.4 * effects[1] + 0.2).abs() < 1e-14);
}

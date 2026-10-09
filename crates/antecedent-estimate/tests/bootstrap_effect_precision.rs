//! Full original nonlinear mediation and latent-mixture bootstrap SE diagnostics.
//! Mediation: randomized A, X uniform{-1,0,1,2,3}, M=0.5A+0.5X+0.8Z,
//! Y=1+0.8A+0.6M+0.4M²+0.3AM+0.7X+0.5E, independent standard normal Z,E.
//! Exact natural direct/indirect/total effects are [0.95,0.75,1.7].
//! Mixture: iid class probability(0.6,0.4), randomized A, normal noiseSD.5;
//! class means A and20-2A, canonical effects[-2,1], weights[0.4,0.6], mean-0.2.
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
fn mixture(n: usize, seed: u64) -> Option<Fit> {
    let mut rng = candidate::Generator::new(seed);
    let (mut a, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let av = rng.binary(0.5) as f64;
        let second = rng.binary(0.4) == 1;
        a.push(av);
        y.push(if second { 20. - 2. * av } else { av } + 0.5 * rng.normal());
    }
    let data = LatentClassData { treatment: &a, outcome: &y, covariates: &[] };
    let mut config = LatentClassConfig::new(2, seed).declare_conditional_randomization();
    config.bootstrap_replicates = 300;
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

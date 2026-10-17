//! Numerical controls on the posterior draws behind a Bayesian linear-adjustment interval:
//! a supported minimum draw count, the endpoint Monte Carlo error reported with the interval
//! and bounded, and the actual draws and seed in the replay identity.
//!
//! The data are the linear Gaussian SCM `bayesian_linear_scm` (1,000 rows, effect 0.4) at
//! the seed each acceptance case names.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

mod common;

use antecedent::{
    BayesianConfig, CausalError, InferenceMode, PosteriorDrawControls, RefuteSuite, Study,
    StudyResult,
};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::count_fitted_posterior::MAX_ENDPOINT_MCSE_FRACTION;
use antecedent_estimate::{
    BayesianBackendKind, MIN_CONTROLLED_DRAWS, endpoint_mcse_from_notes, supported_min_draws,
};
use antecedent_graph::{Dag, DenseNodeId};
use common::calibration::gaussian;
use common::static_dgp::bayesian_linear_scm;

const EFFECT: f64 = 0.4;
const ROWS: usize = 1000;

fn dag(edges: &[(u32, u32)], nodes: u32) -> Dag {
    let mut g = Dag::with_variables(nodes);
    for &(a, b) in edges {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    g
}

/// Columns `t, y, z`: `z → t`, `z → y`, `t → y`.
fn linear_dag() -> Dag {
    dag(&[(2, 0), (2, 1), (0, 1)], 3)
}

/// An inference mode and whether the study asks for the posterior draw controls.
struct Setup {
    inference: InferenceMode,
    controls: bool,
}

impl From<InferenceMode> for Setup {
    fn from(inference: InferenceMode) -> Self {
        Self { inference, controls: false }
    }
}

fn controlled(config: BayesianConfig) -> Setup {
    Setup { inference: InferenceMode::Bayesian(config), controls: true }
}

fn build(data: TabularData, graph: Dag, setup: impl Into<Setup>) -> Result<Study, CausalError> {
    let Setup { inference, controls } = setup.into();
    let builder = Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .inference(inference)
        .refute(RefuteSuite::None);
    let builder =
        if controls { builder.draw_controls(PosteriorDrawControls::Measured) } else { builder };
    builder.build()
}

fn refusal(result: Result<Study, CausalError>) -> String {
    result.expect_err("the study is refused").to_string()
}

fn effect_draws(result: &StudyResult) -> Vec<f64> {
    let posterior = result.posterior.as_ref().expect("a posterior");
    posterior.draws.column(posterior.effect_column().expect("effect column")).unwrap().to_vec()
}

/// Half the spread of sorted draws between the ranks one binomial standard deviation either
/// side of `m p`: the distribution-free standard error of the `p`-quantile, coded here
/// independently of the library.
fn order_statistic_mcse(sorted: &[f64], p: f64) -> f64 {
    let m = sorted.len() as f64;
    let spread = (m * p * (1.0 - p)).sqrt();
    let rank = |h: f64| h.clamp(0.0, m - 1.0) as usize;
    (sorted[rank((m * p + spread).ceil())] - sorted[rank((m * p - spread).floor())]) / 2.0
}

#[test]
fn a_draw_count_below_the_supported_minimum_is_refused_naming_it() {
    let data = bayesian_linear_scm(ROWS, EFFECT, 2_969_080);
    assert_eq!(MIN_CONTROLLED_DRAWS, 2000);
    assert_eq!(supported_min_draws(BayesianBackendKind::Laplace), 2000);
    assert_eq!(supported_min_draws(BayesianBackendKind::ConjugateGaussian), 2000);
    assert_eq!(supported_min_draws(BayesianBackendKind::Hmc), 3000);

    for config in [BayesianConfig::laplace(), BayesianConfig::conjugate()] {
        for draws in [200, 1999] {
            let message = refusal(build(
                data.clone(),
                linear_dag(),
                controlled(config.clone().n_draws(draws)),
            ));
            assert!(message.contains("reason=posterior_draws_below_supported"), "{message}");
            assert!(message.contains("supported minimum is 2000"), "{message}");
        }
        assert!(
            build(data.clone(), linear_dag(), controlled(config.clone().n_draws(2000))).is_ok()
        );
        // A default draw count is below the minimum too: the minimum is asked for, not assumed.
        let message = refusal(build(data.clone(), linear_dag(), controlled(config)));
        assert!(message.contains("reason=posterior_draws_below_supported"), "{message}");
    }

    // HMC needs the draw floor its chain diagnostics need, and says so.
    for draws in [200, 2000, 2999] {
        let message = refusal(build(
            data.clone(),
            linear_dag(),
            controlled(BayesianConfig::hmc().n_draws(draws)),
        ));
        assert!(message.contains("reason=posterior_draws_below_supported"), "{message}");
        assert!(message.contains("supported minimum is 3000"), "{message}");
    }
    assert!(
        build(data.clone(), linear_dag(), controlled(BayesianConfig::hmc().n_draws(3000))).is_ok()
    );

    // Released behavior is unchanged without the controls: a 200-draw study runs as asked.
    let plain = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200));
    let result = build(data, linear_dag(), plain)
        .unwrap()
        .run(&ExecutionContext::for_tests(2_969_080))
        .unwrap();
    assert_eq!(effect_draws(&result).len(), 200);
    assert!(
        endpoint_mcse_from_notes(&result.posterior.as_ref().unwrap().diagnostics.notes).is_none()
    );
}

#[test]
fn supplied_draws_give_an_interval_with_reported_endpoint_monte_carlo_error() {
    for (draws, seed) in
        [(2000, 3_272_793), (3000, 1_520_647), (4000, 5_482_399), (8000, 4_550_226)]
    {
        let data = bayesian_linear_scm(ROWS, EFFECT, seed);
        let study = build(data, linear_dag(), controlled(BayesianConfig::laplace().n_draws(draws)))
            .unwrap();
        let result = study.run(&ExecutionContext::for_tests(seed)).unwrap();
        let posterior = result.posterior.as_ref().unwrap();
        assert_eq!(posterior.draws.n_draws, draws, "the interval is from the supplied draws");

        let reported = endpoint_mcse_from_notes(&posterior.diagnostics.notes)
            .unwrap_or_else(|| panic!("{draws} draws: endpoint error not reported"));
        assert_eq!(reported.draws, draws);
        assert_eq!(reported.effective_draws.to_bits(), (draws as f64).to_bits());
        assert!(
            result.diagnostics.iter().any(|d| d.code.as_ref() == "estimate.bayesian.draw_controls")
        );

        // The reported errors are the order-statistic errors of the effect draws, computed here
        // independently, relative to the published interval.
        let mut sorted = effect_draws(&result);
        sorted.sort_by(f64::total_cmp);
        let column = posterior.effect_column().unwrap();
        let (lower, upper) = (posterior.summaries.q025[column], posterior.summaries.q975[column]);
        let (lower_mcse, upper_mcse) =
            (order_statistic_mcse(&sorted, 0.025), order_statistic_mcse(&sorted, 0.975));
        assert!((reported.lower_mcse - lower_mcse).abs() <= 1e-12 * lower_mcse.max(1e-300));
        assert!((reported.upper_mcse - upper_mcse).abs() <= 1e-12 * upper_mcse.max(1e-300));
        let fraction = lower_mcse.max(upper_mcse) / (upper - lower);
        assert!((reported.width_fraction - fraction).abs() <= 1e-12 * fraction);
        assert!(fraction < MAX_ENDPOINT_MCSE_FRACTION, "{draws} draws: {fraction}");

        // Known truth for the size of that error: the posterior of the effect is close to
        // normal with standard deviation `s`, so the 2.5% quantile of `m` independent draws
        // has standard error `s sqrt(p (1 - p) / m) / phi(z_p)`.
        let s = posterior.summaries.sd[column];
        let phi =
            (-0.5 * 1.959_963_984_540_054_f64.powi(2)).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let expected = s * (0.025_f64 * 0.975 / draws as f64).sqrt() / phi;
        // The order-statistic error is itself an estimate from about 2 sqrt(m p (1 - p))
        // ranks: over 20,000 simulated normal samples its ratio to the formula has 0.05% and
        // 99.95% quantiles of 0.35 and 2.09 at 2,000 draws (0.43 and 2.06 at 3,000, 0.49 and
        // 1.73 at 8,000), so the band is those extremes.
        let ratio = reported.lower_mcse / expected;
        assert!((0.35..2.1).contains(&ratio), "{draws} draws: {ratio}");

        // More draws shrink it: the error is the draws', not the posterior's.
        if draws == 8000 {
            let fewer = build(
                bayesian_linear_scm(ROWS, EFFECT, seed),
                linear_dag(),
                controlled(BayesianConfig::laplace().n_draws(2000)),
            )
            .unwrap()
            .run(&ExecutionContext::for_tests(seed))
            .unwrap();
            let fewer =
                endpoint_mcse_from_notes(&fewer.posterior.as_ref().unwrap().diagnostics.notes)
                    .unwrap();
            assert!(reported.width_fraction < fewer.width_fraction);
        }
    }
}

/// The generator's data plus a near-copy `z2 = z + 1e-6 e` of the confounder, also a cause of
/// treatment and outcome: both are adjusted for, so the posterior of their two coefficients
/// is a ridge the data barely constrain along one direction.
fn with_collinear_confounder(data: &TabularData, seed: u64) -> TabularData {
    let column = |name: &str| data.float64_values(data.schema().id_of(name).unwrap()).unwrap();
    let (t, y, z) = (column("t"), column("y"), column("z"));
    let mut noise = gaussian(seed ^ 0x5EED);
    let z2: Vec<f64> = z.iter().map(|value| value + 1e-6 * noise()).collect();
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("z2", z2.as_slice()),
    ])
    .unwrap()
}

#[test]
fn a_sampler_that_mixes_poorly_is_refused_naming_the_chain_diagnostics() {
    let seed = 8_863_563;
    let data = with_collinear_confounder(&bayesian_linear_scm(ROWS, EFFECT, seed), seed);
    let graph = || dag(&[(2, 0), (2, 1), (3, 0), (3, 1), (0, 1)], 4);
    let ctx = ExecutionContext::for_tests(seed);

    // Expected ESS order: the data pin the coefficients' sum to sd ~0.02 (precision ~ n var(z) /
    // sigma^2) but leave their difference to the N(0, 10^2) prior, so the posterior covariance
    // has condition ~ (10 / 0.02)^2 ~ 2.5e5; both marginal sds are alike, so a diagonal metric
    // cannot rescale it away. The leapfrog step is bound by the narrow direction (~0.02), a
    // 16-step trajectory covers ~0.3, and crossing the wide direction by a random walk of
    // trajectories takes ~(10 / 0.3)^2 ~ 1e3 iterations per effective draw: bulk and tail ESS
    // of order 10 over 4 chains x 3,000 draws against the 400 floor, and split R-hat well
    // above 1.01 because chains barely leave their starts.
    // HMC with a diagonal metric cannot traverse the ridge in 3,000 draws: its chains stay
    // correlated and disagree, and the control refuses naming R-hat and the effective sample
    // sizes it measured rather than publishing an interval the draws, not the posterior, set.
    let hmc =
        build(data.clone(), graph(), controlled(BayesianConfig::hmc().n_draws(3000))).unwrap();
    let message = hmc.run(&ctx).expect_err("poor mixing is refused").to_string();
    assert!(message.contains("reason=posterior_draws_not_converged"), "{message}");
    assert!(message.contains("rhat="), "{message}");
    assert!(message.contains("ess_bulk=") && message.contains("ess_tail="), "{message}");

    // The same data are no obstacle to an exact sampler, whose independent draws pass the
    // control: the refusal is about mixing, not the design.
    let exact = build(data, graph(), controlled(BayesianConfig::laplace().n_draws(3000))).unwrap();
    let result = exact.run(&ctx).unwrap();
    let reported = endpoint_mcse_from_notes(&result.posterior.as_ref().unwrap().diagnostics.notes)
        .expect("the exact sampler's endpoint error is reported");
    assert!(reported.width_fraction < MAX_ENDPOINT_MCSE_FRACTION);
}

#[test]
fn draws_and_seed_are_in_the_replay_identity_and_replay_reproduces_the_draws() {
    let seed = 9_483_778;
    let data = bayesian_linear_scm(ROWS, EFFECT, seed);
    let ctx = ExecutionContext::for_tests(seed);
    let study = |draws: usize, controls: bool| {
        let config = BayesianConfig::conjugate().n_draws(draws);
        let inference =
            if controls { controlled(config) } else { InferenceMode::Bayesian(config).into() };
        build(data.clone(), linear_dag(), inference).unwrap()
    };
    let binding = |study: &Study| study.inspect().unwrap().identities.inference_binding;

    // Changing the draw count changes the identity; so does asking for the controls.
    let base = binding(&study(3000, true));
    assert_eq!(base, binding(&study(3000, true)));
    assert_ne!(base, binding(&study(4000, true)));
    assert_ne!(base, binding(&study(3000, false)));

    // Changing the seed changes the execution identity, which the claim carries.
    let execution = |seed: u64| {
        antecedent_io::execution_digest(&antecedent_io::execution_identity_from_context(
            &ExecutionContext::for_tests(seed),
        ))
        .unwrap()
    };
    assert_eq!(execution(seed), execution(seed));
    assert_ne!(execution(seed), execution(seed + 1));
    let prepared = study(3000, true).prepare(&ctx).unwrap();
    let contract = prepared.contract().unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();
    let other_ctx = ExecutionContext::for_tests(seed + 1);
    let other = prepared.estimate(&data, &other_ctx).unwrap();
    let claim = |result: &StudyResult, ctx: &ExecutionContext| {
        result.claim(&contract, ctx).unwrap().execution
    };
    assert_ne!(claim(&first, &ctx), claim(&other, &other_ctx));

    // Replay with the same draws and seed reproduces every draw bit for bit; another seed
    // does not.
    let again = prepared.estimate(&data, &ExecutionContext::for_tests(seed)).unwrap();
    let (draws, replayed) = (effect_draws(&first), effect_draws(&again));
    assert_eq!(draws.len(), 3000);
    assert!(draws.iter().zip(&replayed).all(|(a, b)| a.to_bits() == b.to_bits()));
    assert!(draws.iter().zip(effect_draws(&other)).any(|(a, b)| a.to_bits() != b.to_bits()));

    // The exported artifact states the actual draws, the controls and the seed.
    for controls in [true, false] {
        let prepared = study(3000, controls).prepare(&ctx).unwrap();
        let result = prepared.estimate(&data, &ctx).unwrap();
        let bytes = prepared.encode_contracted_result(&result, "draws", &ctx).unwrap();
        let section = antecedent_io::consume_analysis_result(&bytes).unwrap().contract.unwrap();
        let bayesian = section.inference_binding.unwrap().bayesian.unwrap();
        assert_eq!(bayesian.n_draws, 3000);
        assert_eq!(bayesian.draw_controls.as_deref(), controls.then_some("measured"));
        assert_eq!(section.execution.unwrap().seed, seed);
    }
}

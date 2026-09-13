//! 1.9 coverage of licensed temporal / mixture intervals.
//!
//! Ignored tests run via `scripts/gate_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]

use std::sync::Arc;

use antecedent::discovery::GraphPosterior;
use antecedent::validate::PredictiveCheckKind;
use antecedent::{BayesianConfig, ClassPrior, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Lag, MediationContrast,
    MediationQuery, ResponseFunctional, ResponseQuery, ResponseUncertainty, TemporalEffectQuery,
    TemporalPolicy, TemporalResponseSpec, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalCpdag, TemporalDag, TemporalPag, ensure_lagged};
use antecedent_identify::IdentificationStatus;
use antecedent_prob::InferenceDiagnostics;

const Z90: f64 = 1.644_853_626_951_472_2;
const TRUTH: f64 = 0.8;
const N_SIM: u32 = 40;
const N: usize = 160;
const BOOT: u32 = 24;
const DRAWS: usize = 160;

fn coverage_ok(name: &str, covered: u32, n_sim: u32, level: f64) {
    let rate = f64::from(covered) / f64::from(n_sim);
    let se = (level * (1.0 - level) / f64::from(n_sim)).sqrt();
    let lo = (level - 4.0 * se).max(0.70);
    let hi = (level + 4.0 * se).min(1.0);
    assert!(
        rate >= lo && rate <= hi,
        "{name} {:.0}% coverage={rate:.3} outside [{lo:.3}, {hi:.3}] ({covered}/{n_sim})",
        level * 100.0
    );
}

fn freq_covers(ate: f64, se: Option<f64>, truth: f64) -> bool {
    let Some(se) = se.filter(|se| se.is_finite() && *se > 0.0) else {
        return false;
    };
    (truth - ate).abs() <= Z90 * se
}

fn posterior_covers(result: &antecedent::StudyResult, truth: f64, level: f64) -> bool {
    let Some(post) = result.posterior.as_ref() else {
        return false;
    };
    let Some(col) = post.effect_column() else {
        return false;
    };
    let Ok(draws) = post.draws.column(col) else {
        return false;
    };
    let mut values = draws.to_vec();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let last = (values.len() - 1) as f64;
    let lo_p = (1.0 - level) / 2.0;
    let hi_p = 1.0 - lo_p;
    let at = |q: f64| values[(last * q).round().clamp(0.0, last) as usize];
    truth >= at(lo_p) && truth <= at(hi_p)
}

fn box_muller(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed | 1;
    move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let u1 = ((state >> 33) as f64 / (1u64 << 31) as f64).clamp(1e-12, 1.0);
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let u2 = (state >> 33) as f64 / (1u64 << 31) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn noisy_xy(n: usize, seed: u64) -> TimeSeriesData {
    let mut gauss = box_muller(seed);
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = 0.4 * gauss();
        y[t] = TRUTH * x[t - 1] + 0.35 * gauss();
    }
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

fn noisy_xyz(n: usize, seed: u64) -> TimeSeriesData {
    let mut gauss = box_muller(seed);
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    for i in 1..n {
        z[i] = gauss();
        t[i] = 0.4 * z[i] + 0.5 * gauss();
        y[i] = TRUTH * t[i - 1] + 0.35 * gauss();
    }
    TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap()
}

fn noisy_tmy(n: usize, seed: u64) -> TimeSeriesData {
    let mut gauss = box_muller(seed);
    let mut t = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 1..n {
        t[i] = 0.5 * gauss();
        m[i] = 0.6 * t[i - 1] + 0.25 * gauss();
        y[i] = 0.4 * t[i - 1] + 0.5 * m[i] + 0.25 * gauss();
    }
    TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())],
        1,
    )
    .unwrap()
}

fn pulse_query() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
}

fn single_sustained() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -1, 1.0)
        .with_policy(TemporalPolicy::sustained(-1, -1))
        .with_horizon_steps(1)
}

fn multi_sustained() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -2, 1.0)
        .with_policy(TemporalPolicy::sustained(-2, -1))
        .with_horizon_steps(1)
}

fn xy_dag() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(x1, y0).unwrap();
    g
}

fn xyz_cpdag() -> TemporalCpdag {
    let mut g = TemporalCpdag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_undirected(z1, t1).unwrap();
    g
}

fn directed_pag() -> TemporalPag {
    let mut g = TemporalPag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    g.insert_directed(z1, t1).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g
}

fn mediation_cpdag() -> TemporalCpdag {
    let mut g = TemporalCpdag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let m0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(t1, m0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(m0, y0).unwrap();
    g
}

fn identified_plus_unidentified_dbn() -> GraphPosterior {
    let contemporaneous = 0_u64;
    GraphPosterior::new(
        2,
        vec![0.7, 0.3],
        vec![contemporaneous, contemporaneous],
        vec![0.0; 4],
        vec![0.0; 4],
        1.0 / (0.7 * 0.7 + 0.3 * 0.3),
        InferenceDiagnostics::analytic("v19_mixture"),
        0,
    )
    .unwrap()
    .with_lagged_marginals(1, vec![0.3, 1.0, 0.0, 0.0])
    .unwrap()
    .with_lag_masks(vec![2, 3])
    .unwrap()
}

fn two_identified_dbn() -> GraphPosterior {
    GraphPosterior::new(
        2,
        vec![0.6, 0.4],
        vec![0, 0],
        vec![0.0; 4],
        vec![0.0; 4],
        1.0 / (0.6 * 0.6 + 0.4 * 0.4),
        InferenceDiagnostics::analytic("v19_two_identified"),
        0,
    )
    .unwrap()
    .with_lagged_marginals(1, vec![0.0, 1.0, 0.0, 0.0])
    .unwrap()
    .with_lag_masks(vec![2, 2])
    .unwrap()
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(8.0))
}

fn run_freq_dbn(
    data: TimeSeriesData,
    query: TemporalEffectQuery,
    gp: GraphPosterior,
) -> antecedent::StudyResult {
    Study::series(data)
        .graph_posterior(gp)
        .temporal_query(query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(BOOT)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(41))
        .unwrap()
}

fn run_freq_class(
    data: TimeSeriesData,
    graph: impl Into<antecedent::AcceptedGraph>,
    query: CausalQuery,
) -> antecedent::StudyResult {
    Study::series(data)
        .graph(graph.into())
        .query(query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(BOOT)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(17))
        .unwrap()
}

fn run_bayes(
    data: TimeSeriesData,
    graph: impl Into<antecedent::AcceptedGraph>,
    query: CausalQuery,
    prior: Option<ClassPrior>,
    suite: RefuteSuite,
) -> antecedent::StudyResult {
    let mut builder = Study::series(data).graph(graph.into()).query(query).inference(bayes());
    if let Some(prior) = prior {
        builder = builder.class_prior(prior);
    }
    builder
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(21))
        .unwrap()
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_dbn_pulse_shared_block_nominal_90_coverage() {
    let mut covered = 0;
    for s in 0..N_SIM {
        let result =
            run_freq_dbn(noisy_xy(N, 9_000 + u64::from(s)), pulse_query(), two_identified_dbn());
        if freq_covers(result.estimate.ate, result.estimate.se_bootstrap, TRUTH) {
            covered += 1;
        }
    }
    coverage_ok("frequentist DBN Pulse", covered, N_SIM, 0.9);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_dbn_sustained_shared_block_nominal_90_coverage() {
    let mut covered = 0;
    for s in 0..N_SIM {
        let result = run_freq_dbn(
            noisy_xy(N, 10_000 + u64::from(s)),
            single_sustained(),
            two_identified_dbn(),
        );
        if freq_covers(result.estimate.ate, result.estimate.se_bootstrap, TRUTH) {
            covered += 1;
        }
    }
    coverage_ok("frequentist DBN single-step Sustained", covered, N_SIM, 0.9);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_dbn_multistep_sustained_shared_block_nominal_90_coverage() {
    let mut covered = 0;
    for s in 0..N_SIM {
        let result = run_freq_dbn(
            noisy_xy(N, 11_000 + u64::from(s)),
            multi_sustained(),
            two_identified_dbn(),
        );
        if freq_covers(result.estimate.ate, result.estimate.se_bootstrap, TRUTH) {
            covered += 1;
        }
    }
    coverage_ok("frequentist DBN multi-step Sustained", covered, N_SIM, 0.9);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_cpdag_pulse_envelope_nominal_90_coverage() {
    let mut covered = 0;
    for s in 0..N_SIM {
        let result = run_freq_class(
            noisy_xyz(N, 12_000 + u64::from(s)),
            xyz_cpdag(),
            CausalQuery::TemporalEffect(pulse_query()),
        );
        assert!(
            result
                .diagnostics
                .iter()
                .any(|d| d.code.as_ref() == "estimate.temporal_class.frequentist.shared_block"),
            "class envelope must publish shared-block SE"
        );
        if freq_covers(result.estimate.ate, result.estimate.se_bootstrap, TRUTH) {
            covered += 1;
        }
    }
    coverage_ok("frequentist TemporalCpdag Pulse", covered, N_SIM, 0.9);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_cpdag_sustained_envelope_nominal_90_coverage() {
    let mut covered = 0;
    for s in 0..N_SIM {
        let result = run_freq_class(
            noisy_xyz(N, 13_000 + u64::from(s)),
            xyz_cpdag(),
            CausalQuery::TemporalEffect(single_sustained()),
        );
        if freq_covers(result.estimate.ate, result.estimate.se_bootstrap, TRUTH) {
            covered += 1;
        }
    }
    coverage_ok("frequentist TemporalCpdag Sustained", covered, N_SIM, 0.9);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_pag_pulse_envelope_nominal_90_coverage() {
    let mut covered = 0;
    for s in 0..N_SIM {
        let result = run_freq_class(
            noisy_xyz(N, 14_000 + u64::from(s)),
            directed_pag(),
            CausalQuery::TemporalEffect(pulse_query()),
        );
        if freq_covers(
            result.estimate.ate,
            result.estimate.se_bootstrap.or_else(|| {
                result.estimate.se_analytic.is_finite().then_some(result.estimate.se_analytic)
            }),
            TRUTH,
        ) {
            covered += 1;
        }
    }
    coverage_ok("frequentist TemporalPag Pulse", covered, N_SIM, 0.9);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_dag_multistep_sustained_nominal_90_coverage() {
    let mut covered = 0;
    for s in 0..N_SIM {
        let result = run_bayes(
            noisy_xy(N, 15_000 + u64::from(s)),
            xy_dag(),
            CausalQuery::TemporalEffect(multi_sustained()),
            None,
            RefuteSuite::None,
        );
        if posterior_covers(&result, TRUTH, 0.9) {
            covered += 1;
        }
    }
    coverage_ok("Bayesian TemporalDag multi-step Sustained", covered, N_SIM, 0.9);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_dag_response_curve_nominal_90_coverage() {
    let mut covered = 0;
    let query = CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.0, 1.0])),
            ),
        })
        .with_temporal(
            TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap(),
        ),
    );
    for s in 0..N_SIM {
        let result = run_bayes(
            noisy_xy(N, 16_000 + u64::from(s)),
            xy_dag(),
            query.clone(),
            None,
            RefuteSuite::None,
        );
        let response = result.response.as_ref().expect("curve");
        let covered_now = match &response.uncertainty {
            ResponseUncertainty::PointwiseBand { lower, upper, .. } => {
                lower.len() >= 2 && upper.len() >= 2 && TRUTH >= lower[1] && TRUTH <= upper[1]
            }
            _ => posterior_covers(&result, TRUTH, 0.9),
        };
        if covered_now {
            covered += 1;
        }
    }
    coverage_ok("Bayesian TemporalDag ResponseCurve", covered, N_SIM, 0.9);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_cpdag_pulse_class_prior_nominal_90_coverage() {
    let mut covered = 0;
    let prior = ClassPrior::from_ordered([0.5, 0.5]).unwrap();
    for s in 0..N_SIM {
        let result = run_bayes(
            noisy_xyz(N, 17_000 + u64::from(s)),
            xyz_cpdag(),
            CausalQuery::TemporalEffect(pulse_query()),
            Some(prior.clone()),
            RefuteSuite::None,
        );
        if posterior_covers(&result, TRUTH, 0.9) {
            covered += 1;
        }
    }
    coverage_ok("Bayesian TemporalCpdag Pulse class-prior", covered, N_SIM, 0.9);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_cpdag_sustained_class_prior_nominal_90_coverage() {
    let mut covered = 0;
    let prior = ClassPrior::from_ordered([0.5, 0.5]).unwrap();
    for s in 0..N_SIM {
        let result = run_bayes(
            noisy_xyz(N, 18_000 + u64::from(s)),
            xyz_cpdag(),
            CausalQuery::TemporalEffect(single_sustained()),
            Some(prior.clone()),
            RefuteSuite::None,
        );
        if posterior_covers(&result, TRUTH, 0.9) {
            covered += 1;
        }
    }
    coverage_ok("Bayesian TemporalCpdag Sustained class-prior", covered, N_SIM, 0.9);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_pag_pulse_nominal_90_coverage() {
    let mut covered = 0;
    for s in 0..N_SIM {
        let result = run_bayes(
            noisy_xyz(N, 19_000 + u64::from(s)),
            directed_pag(),
            CausalQuery::TemporalEffect(pulse_query()),
            Some(ClassPrior::from_ordered([1.0]).unwrap()),
            RefuteSuite::None,
        );
        if posterior_covers(&result, TRUTH, 0.9) {
            covered += 1;
        }
    }
    coverage_ok("Bayesian TemporalPag Pulse", covered, N_SIM, 0.9);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_cpdag_mediation_envelope_nominal_90_coverage() {
    let mut covered = 0;
    let query = CausalQuery::Mediation(
        MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        )
        .with_horizons(vec![1])
        .unwrap(),
    );
    for s in 0..N_SIM {
        let result = run_bayes(
            noisy_tmy(N, 20_000 + u64::from(s)),
            mediation_cpdag(),
            query.clone(),
            None,
            RefuteSuite::None,
        );
        let truth_nie = 0.3;
        let covered_now = result.posterior.as_ref().map_or_else(
            || {
                result.structural_response.as_ref().is_some_and(|mixture| {
                    mixture.atoms.iter().filter_map(|atom| atom.posterior.as_ref()).any(|post| {
                        post.effect_column().is_some_and(|col| {
                            post.draws.column(col).ok().is_some_and(|draws| {
                                let mut values = draws.to_vec();
                                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                let last = (values.len() - 1) as f64;
                                let lo = values[(last * 0.05).round() as usize];
                                let hi = values[(last * 0.95).round() as usize];
                                truth_nie >= lo && truth_nie <= hi
                            })
                        })
                    })
                })
            },
            |_| posterior_covers(&result, truth_nie, 0.9),
        );
        if covered_now {
            covered += 1;
        }
    }
    coverage_ok("Bayesian TemporalCpdag mediation", covered, N_SIM, 0.9);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn dbn_mixture_functional_retains_unidentified_mass() {
    let mut covered = 0;
    for s in 0..N_SIM {
        let result = run_freq_dbn(
            noisy_xy(N, 21_000 + u64::from(s)),
            pulse_query(),
            identified_plus_unidentified_dbn(),
        );
        let unidentified = result.posterior.as_ref().map_or(0.3, |post| post.unidentified_mass);
        assert!(
            (unidentified - 0.3).abs() < 1e-9
                || result.identification.status == IdentificationStatus::GraphDependent,
            "unidentified mass must stay a separate axis, got {unidentified}"
        );
        if freq_covers(result.estimate.ate, result.estimate.se_bootstrap, TRUTH) {
            covered += 1;
        }
    }
    coverage_ok("DBN mixture functional E[τ|identified]", covered, N_SIM, 0.9);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn class_prior_mixture_functional_nominal_90_coverage() {
    let mut covered = 0;
    let prior = ClassPrior::from_ordered([0.3, 0.7]).unwrap();
    for s in 0..N_SIM {
        let result = run_bayes(
            noisy_xyz(N, 22_000 + u64::from(s)),
            xyz_cpdag(),
            CausalQuery::TemporalEffect(pulse_query()),
            Some(prior.clone()),
            RefuteSuite::None,
        );
        if let Some(structural) = result.structural_response.as_ref() {
            assert!(
                structural.unidentified_mass >= 0.0,
                "class mixture must retain unidentified mass as a separate axis"
            );
        }
        if posterior_covers(&result, TRUTH, 0.9) {
            covered += 1;
        }
    }
    coverage_ok("class-prior mixture functional", covered, N_SIM, 0.9);
}

#[test]
fn bayesian_full_pins_numeric_ppc_summary() {
    let result = run_bayes(
        noisy_xy(N, 7),
        xy_dag(),
        CausalQuery::TemporalEffect(pulse_query()),
        None,
        RefuteSuite::Full,
    );
    assert!(
        result.predictive_checks.iter().any(|c| c.kind == PredictiveCheckKind::Prior),
        "full must run prior PPC"
    );
    let ppc = result
        .predictive_checks
        .iter()
        .find(|c| c.kind == PredictiveCheckKind::Posterior)
        .expect("full must run posterior PPC");
    assert!(result.posterior.as_ref().and_then(|p| p.prior_sensitivity.as_ref()).is_some());
    assert!(ppc.p_value.is_finite());
    assert!(ppc.predictive_mean.is_finite());
    assert!(ppc.predictive_sd.is_finite());
    assert!(ppc.n_sims >= 64, "PPC must run a numeric simulation, n_sims={}", ppc.n_sims);
    assert!(
        (ppc.predictive_mean - ppc.observed).abs() < 0.35,
        "numeric PPC predictive_mean={} observed={}",
        ppc.predictive_mean,
        ppc.observed
    );
    assert!(
        ppc.p_value > 0.01,
        "well-specified conjugate DGP must not reject posterior PPC (p={})",
        ppc.p_value
    );
}

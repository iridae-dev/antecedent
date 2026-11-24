//! Twin equality of the hidden `discover_*` engines and the compiled discovery routes.
//!
//! Each test runs the direct engine and the `DiscoverStatic` / `DiscoverTemporal` study that
//! replaces it on the same data and parameters, and asserts that the discovered structure is the
//! same: graphs, scored links and separating sets (compared as their exact `Debug` text, which
//! prints every float to its shortest round-trip form) for the frequentist searches, and
//! `to_bits()` equality of the weights, with equal atoms and algorithm tag, for the posteriors.
//! The compiled route calls the same engine function with the same parameters, so any
//! difference is a bug.
//!
//! Seeds: the compiled route hands the engine the context it was run with, so the direct call
//! and the study share one seed here (`ExecutionContext::for_tests` with the same value); no
//! child seed is derived. A study run under another context seed draws different chains, as a
//! direct call under that seed does.
//!
//! The pooled engines run through the accepted-posterior pooling route, which is covered by the
//! structure prior route tests; the one pooled twin here pins the pooling of exact posteriors.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(deprecated, reason = "the direct discovery engines are the paths under comparison")]
#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "integration test with deterministic fixtures and bitwise comparisons"
)]

use std::sync::Arc;

use antecedent::discovery::{
    BayesianDiscoverParams, DiscoverParams, GraphMcmcSchedule, GraphPosterior, StaticDiscoverParams,
};
use antecedent::discovery_defaults::resolve_ci;
use antecedent::{CompiledExecution, DiscoveredStructure, DiscoverySpec, Study};
use antecedent_core::{CausalRng, ExecutionContext, VariableId};
use antecedent_data::TableView;
use antecedent_data::{
    MultiEnvironmentData, SamplingRegularity, TabularData, TimeIndex, TimeSeriesData,
};
use antecedent_discovery::{
    BackgroundKnowledge, DiscoveryResult, MultiDatasetConstraints, two_regime_half_split,
};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn three() -> Vec<VariableId> {
    vec![v(0), v(1), v(2)]
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x5EED_0041)
}

fn on_large_stack(name: &str, body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.into())
        .stack_size(32 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

/// `z -> t -> y`, `z -> y` with uniform (non-Gaussian) noise.
fn chain_table(n: usize, seed: u64) -> TabularData {
    let mut rng = CausalRng::from_seed(seed);
    let mut uniform = || 2.0 * rng.next_f64() - 1.0;
    let (mut z, mut t, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let zi = uniform();
        let ti = 0.8 * zi + uniform();
        let yi = 1.5 * ti + 0.6 * zi + uniform();
        z.push(zi);
        t.push(ti);
        y.push(yi);
    }
    TabularData::from_f64_columns([("z", z.as_slice()), ("t", t.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

fn lag1_series(n: usize, seed: u64) -> TimeSeriesData {
    let mut rng = CausalRng::from_seed(seed);
    let mut normal = || {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = normal();
        y[t] = 0.8 * x[t - 1] + 0.2 * normal();
    }
    let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
    TimeSeriesData::try_new(
        data.storage().clone(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

fn static_params() -> StaticDiscoverParams {
    StaticDiscoverParams {
        alpha: 0.05,
        max_cond_size: 2,
        fdr: None,
        ci: resolve_ci("parcorr", None).unwrap(),
        screen_pc: false,
        max_subset: None,
        knowledge: BackgroundKnowledge::default(),
    }
}

fn temporal_params(max_lag: u32) -> DiscoverParams {
    DiscoverParams {
        max_lag,
        alpha: 0.05,
        fdr: None,
        ci: resolve_ci("parcorr", None).unwrap(),
        multi_dataset: MultiDatasetConstraints::default(),
        max_cond_size: 2,
        knowledge: BackgroundKnowledge::default(),
    }
}

const SCHEDULE: GraphMcmcSchedule =
    GraphMcmcSchedule { n_chains: 2, n_warmup: 60, n_draws: 200, thin: 1 };

/// Run a study through the compiler and return its stamped execution.
fn run(study: &Study) -> CompiledExecution {
    CompiledExecution::run(study, &ctx()).unwrap()
}

fn structure(execution: &CompiledExecution) -> DiscoveredStructure {
    execution.result().discovery().map(|payload| payload.structure.clone()).expect("a discovery")
}

/// The graph, its scored links and its separating sets; separating sets live in a hash map, so
/// they are compared as sorted entries.
fn graph_and_links<G: std::fmt::Debug, R>(
    result: &DiscoveryResult<G, R>,
) -> (String, String, Vec<String>) {
    let mut sepsets: Vec<String> =
        result.sepsets.iter().map(|entry| format!("{entry:?}")).collect();
    sepsets.sort_unstable();
    (format!("{:?}", result.evidence.graph), format!("{:?}", result.evidence.links), sepsets)
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|w| w.to_bits()).collect()
}

fn same_posterior(compiled: &GraphPosterior, direct: &GraphPosterior) {
    assert_eq!(bits(&compiled.weights()[..]), bits(&direct.weights()[..]), "weights");
    assert_eq!(compiled.atoms(), direct.atoms(), "atoms");
    assert_eq!(compiled.algorithm(), direct.algorithm(), "algorithm tag");
}

fn posterior(execution: &CompiledExecution) -> GraphPosterior {
    let DiscoveredStructure::Posterior(found) = structure(execution) else {
        panic!("the search returns a posterior");
    };
    found
}

/// The direct sampler's posterior equals the compiled one, or both refuse it: a sampler that
/// always applies its convergence gate refuses chains that disagree, and the compiled route runs
/// the same engine.
fn same_posterior_or_refusal(
    direct: Result<GraphPosterior, antecedent::CausalError>,
    study: &Study,
) {
    match (direct, CompiledExecution::run(study, &ctx())) {
        (Ok(direct), Ok(compiled)) => same_posterior(&posterior(&compiled), &direct),
        (Err(direct), Err(compiled)) => {
            assert!(direct.to_string().contains("diagnostics gate refused"), "{direct}");
            assert!(compiled.to_string().contains(&direct.to_string()), "{compiled}");
        }
        (direct, compiled) => panic!(
            "the twins disagree: direct {:?}, compiled {:?}",
            direct.map(|_| ()),
            compiled.map(|_| ())
        ),
    }
}

fn static_study(data: &TabularData, spec: DiscoverySpec) -> Study {
    Study::tabular(data.clone()).discovery(spec, three()).build().unwrap()
}

#[test]
fn static_constraint_searches_match_their_engines() {
    on_large_stack("twin-static", || {
        let data = chain_table(400, 3);
        let params = static_params();

        let DiscoveredStructure::StaticCpdag(pc) =
            structure(&run(&static_study(&data, DiscoverySpec::pc(params.clone(), "parcorr"))))
        else {
            panic!("PC returns a static CPDAG");
        };
        let direct = antecedent::discovery::discover_pc(&data, &three(), &params, &ctx()).unwrap();
        assert_eq!(graph_and_links(&pc), graph_and_links(&direct));

        let DiscoveredStructure::StaticPag(fci) =
            structure(&run(&static_study(&data, DiscoverySpec::fci(params.clone(), "parcorr"))))
        else {
            panic!("FCI returns a static PAG");
        };
        let direct = antecedent::discovery::discover_fci(&data, &three(), &params, &ctx()).unwrap();
        assert_eq!(graph_and_links(&fci), graph_and_links(&direct));

        let DiscoveredStructure::StaticPag(rfci) =
            structure(&run(&static_study(&data, DiscoverySpec::rfci(params.clone(), "parcorr"))))
        else {
            panic!("RFCI returns a static PAG");
        };
        let direct =
            antecedent::discovery::discover_rfci(&data, &three(), &params, &ctx()).unwrap();
        assert_eq!(graph_and_links(&rfci), graph_and_links(&direct));

        let DiscoveredStructure::StaticCpdag(ges) =
            structure(&run(&static_study(&data, DiscoverySpec::ges(params.clone(), "parcorr"))))
        else {
            panic!("GES returns a static CPDAG");
        };
        let direct = antecedent::discovery::discover_ges(&data, &three(), &params, &ctx()).unwrap();
        assert_eq!(graph_and_links(&ges), graph_and_links(&direct));

        let DiscoveredStructure::StaticDag(lingam) = structure(&run(&static_study(
            &data,
            DiscoverySpec::lingam(params.clone(), 0.1, "parcorr"),
        ))) else {
            panic!("LiNGAM returns a static DAG");
        };
        let direct =
            antecedent::discovery::discover_lingam(&data, &three(), &params, 0.1, &ctx()).unwrap();
        assert_eq!(graph_and_links(&lingam), graph_and_links(&direct));

        let DiscoveredStructure::Notears(notears) = structure(&run(&static_study(
            &data,
            DiscoverySpec::notears(params.clone(), 0.1, 0.3, true, None, "parcorr"),
        ))) else {
            panic!("NOTEARS returns its weight matrix");
        };
        let direct = antecedent::discovery::discover_notears(
            &data,
            &three(),
            &params,
            0.1,
            0.3,
            true,
            &ctx(),
        )
        .unwrap();
        assert_eq!(graph_and_links(&notears.discovery), graph_and_links(&direct.discovery));
        assert_eq!(bits(&notears.weights), bits(&direct.weights));
        assert_eq!(notears.dim, direct.dim);
    });
}

#[test]
fn temporal_constraint_searches_match_their_engines() {
    on_large_stack("twin-temporal", || {
        let series = lag1_series(200, 11);
        let vars = vec![v(0), v(1)];
        let params = temporal_params(1);
        let temporal = |spec: DiscoverySpec| {
            Study::series(series.clone()).discovery(spec, vars.clone()).build().unwrap()
        };

        let DiscoveredStructure::TemporalDag(pcmci) =
            structure(&run(&temporal(DiscoverySpec::pcmci(params.clone(), "parcorr"))))
        else {
            panic!("PCMCI returns a temporal DAG");
        };
        let direct =
            antecedent::discovery::discover_pcmci(&series, &vars, &params, &ctx()).unwrap();
        assert_eq!(graph_and_links(&pcmci), graph_and_links(&direct));

        let DiscoveredStructure::TemporalCpdag(plus) =
            structure(&run(&temporal(DiscoverySpec::pcmci_plus(params.clone(), "parcorr"))))
        else {
            panic!("PCMCI+ returns a temporal CPDAG");
        };
        let direct =
            antecedent::discovery::discover_pcmci_plus(&series, &vars, &params, &ctx()).unwrap();
        assert_eq!(graph_and_links(&plus), graph_and_links(&direct));

        let DiscoveredStructure::TemporalPag(lpcmci) =
            structure(&run(&temporal(DiscoverySpec::lpcmci(params.clone(), "parcorr"))))
        else {
            panic!("LPCMCI returns a temporal PAG");
        };
        let direct =
            antecedent::discovery::discover_lpcmci(&series, &vars, &params, &ctx()).unwrap();
        assert_eq!(graph_and_links(&lpcmci), graph_and_links(&direct));

        // Regime PCMCI with a supplied half-split assignment.
        let regimes = two_regime_half_split(series.row_count());
        let DiscoveredStructure::Regimes(rpcmci) = structure(&run(&temporal(
            DiscoverySpec::rpcmci(params.clone(), regimes.clone(), Some(40), "parcorr"),
        ))) else {
            panic!("RPCMCI returns per-regime graphs");
        };
        let direct = antecedent::discovery::discover_rpcmci(
            &series,
            &vars,
            &regimes,
            &params,
            Some(40),
            &ctx(),
        )
        .unwrap();
        assert_eq!(format!("{:?}", rpcmci.graphs.graphs), format!("{:?}", direct.graphs.graphs));
        assert_eq!(rpcmci.per_regime.len(), direct.per_regime.len());
        for (compiled, direct) in rpcmci.per_regime.iter().zip(direct.per_regime.iter()) {
            assert_eq!(
                format!("{:?}", compiled.evidence.graph),
                format!("{:?}", direct.evidence.graph)
            );
            assert_eq!(
                format!("{:?}", compiled.evidence.links),
                format!("{:?}", direct.evidence.links)
            );
        }

        // J-PCMCI+ over two environments.
        let environments =
            MultiEnvironmentData::try_new(Arc::from([lag1_series(160, 21), lag1_series(160, 22)]))
                .unwrap();
        let joint = Study::series_multi(environments.clone())
            .discovery(DiscoverySpec::jpcmci_plus(params.clone(), "parcorr"), vars.clone())
            .build()
            .unwrap();
        let DiscoveredStructure::TemporalCpdag(jpcmci) = structure(&run(&joint)) else {
            panic!("J-PCMCI+ returns a temporal CPDAG");
        };
        let direct =
            antecedent::discovery::discover_jpcmci_plus(&environments, &vars, &params, &ctx())
                .unwrap();
        assert_eq!(graph_and_links(&jpcmci), graph_and_links(&direct));

        // PCMCI and PCMCI+ over the units of a panel.
        let units = vec![lag1_series(120, 12), lag1_series(120, 13)];
        let panel = antecedent::panel_of_units(units.clone()).unwrap();
        let panel_study = |spec: DiscoverySpec| {
            Study::panel(panel.clone()).discovery(spec, vars.clone()).build().unwrap()
        };
        let DiscoveredStructure::TemporalDag(on_units) =
            structure(&run(&panel_study(DiscoverySpec::pcmci(params.clone(), "parcorr"))))
        else {
            panic!("PCMCI returns a temporal DAG");
        };
        let direct =
            antecedent::discovery::discover_pcmci_panel(&units, &vars, &params, &ctx()).unwrap();
        assert_eq!(graph_and_links(&on_units), graph_and_links(&direct));
        let DiscoveredStructure::TemporalCpdag(plus_units) =
            structure(&run(&panel_study(DiscoverySpec::pcmci_plus(params.clone(), "parcorr"))))
        else {
            panic!("PCMCI+ returns a temporal CPDAG");
        };
        let direct =
            antecedent::discovery::discover_pcmci_plus_panel(&units, &vars, &params, &ctx())
                .unwrap();
        assert_eq!(graph_and_links(&plus_units), graph_and_links(&direct));
        let DiscoveredStructure::TemporalPag(lpcmci_units) =
            structure(&run(&panel_study(DiscoverySpec::lpcmci(params.clone(), "parcorr"))))
        else {
            panic!("LPCMCI returns a temporal PAG");
        };
        let direct =
            antecedent::discovery::discover_lpcmci_panel(&units, &vars, &params, &ctx()).unwrap();
        assert_eq!(graph_and_links(&lpcmci_units), graph_and_links(&direct));
    });
}

#[test]
fn structure_posteriors_match_their_engines() {
    on_large_stack("twin-posterior", || {
        let data = chain_table(300, 5);
        let params = BayesianDiscoverParams::default();
        let bayesian = |spec: DiscoverySpec| run(&static_study(&data, spec));

        let direct =
            antecedent::discovery::discover_exact_dag_posterior(&data, &three(), &params, &ctx())
                .unwrap();
        same_posterior(
            &posterior(&bayesian(DiscoverySpec::exact_dag_posterior(params.clone()))),
            &direct,
        );

        let direct = antecedent::discovery::discover_order_mcmc(
            &data,
            &three(),
            &params,
            &SCHEDULE,
            false,
            &ctx(),
        )
        .unwrap();
        same_posterior(
            &posterior(&bayesian(DiscoverySpec::order_mcmc(params.clone(), SCHEDULE, false))),
            &direct,
        );

        // Structure MCMC always applies its convergence gate. On this fixture its chains settle
        // in different orientations of Markov-equivalent complete DAGs, so the engine refuses
        // the posterior; the compiled route reaches the same engine and refuses it the same way.
        same_posterior_or_refusal(
            antecedent::discovery::discover_structure_mcmc(
                &data,
                &three(),
                &params,
                &SCHEDULE,
                &ctx(),
            ),
            &static_study(&data, DiscoverySpec::structure_mcmc(params.clone(), SCHEDULE)),
        );

        let screen = static_params();
        same_posterior_or_refusal(
            antecedent::discovery::discover_ci_screened_posterior(
                &data,
                &three(),
                &params,
                &screen,
                &SCHEDULE,
                antecedent::discovery::CiSoftWeight::None,
                &ctx(),
            ),
            &static_study(
                &data,
                DiscoverySpec::ci_screened_posterior(
                    params.clone(),
                    screen.clone(),
                    SCHEDULE,
                    antecedent::discovery::CiSoftWeight::None,
                    "parcorr",
                ),
            ),
        );

        // Edge marginals: the exact ones and the screened ones.
        let DiscoveredStructure::EdgeMarginals(marginals) =
            structure(&bayesian(DiscoverySpec::exact_dag_marginals(params.clone())))
        else {
            panic!("the marginal search returns edge marginals");
        };
        let direct =
            antecedent::discovery::discover_exact_dag_marginals(&data, &three(), &params, &ctx())
                .unwrap();
        assert_eq!(bits(&marginals.edge_marginals), bits(&direct.edge_marginals));
        assert_eq!(marginals.log_evidence.to_bits(), direct.log_evidence.to_bits());

        let focus: Arc<[VariableId]> = Arc::from(vec![v(1)]);
        let DiscoveredStructure::EdgeMarginals(screened) =
            structure(&bayesian(DiscoverySpec::screened_exact_marginals(
                params.clone(),
                screen.clone(),
                focus.clone(),
                3,
                0.8,
                0.2,
                "parcorr",
            )))
        else {
            panic!("the screened search returns edge marginals");
        };
        let direct = antecedent::discovery::discover_screened_exact_marginals(
            &data,
            &three(),
            &focus,
            3,
            &params,
            &screen,
            0.8,
            0.2,
            &ctx(),
        )
        .unwrap();
        assert_eq!(bits(&screened.edge_marginals), bits(&direct.edge_marginals));
        assert_eq!(screened.log_evidence.to_bits(), direct.log_evidence.to_bits());
        assert_eq!(screened.variables, direct.variables);
    });
}

#[test]
fn the_dbn_template_posterior_matches_its_engine() {
    on_large_stack("twin-dbn", || {
        let series = lag1_series(200, 41);
        let vars = vec![v(0), v(1)];
        let params = BayesianDiscoverParams::default();
        let study = Study::series(series.clone())
            .discovery(
                DiscoverySpec::dbn_posterior(params.clone(), 1, false, SCHEDULE),
                vars.clone(),
            )
            .build()
            .unwrap();
        let direct = antecedent::discovery::discover_dbn_posterior(
            &series,
            &vars,
            &params,
            1,
            false,
            &SCHEDULE,
            &ctx(),
        )
        .unwrap();
        same_posterior(&posterior(&run(&study)), &direct);

        let units = vec![lag1_series(120, 42), lag1_series(120, 43)];
        let panel = antecedent::panel_of_units(units.clone()).unwrap();
        let panel_study = Study::panel(panel)
            .discovery(
                DiscoverySpec::dbn_posterior(params.clone(), 1, false, SCHEDULE),
                vars.clone(),
            )
            .build()
            .unwrap();
        let direct = antecedent::discovery::discover_dbn_posterior_panel(
            &units,
            &vars,
            &params,
            1,
            false,
            &SCHEDULE,
            &ctx(),
        )
        .unwrap();
        same_posterior(&posterior(&run(&panel_study)), &direct);
    });
}

/// The pooled exact posterior is the engine's mixture of the per-table exact posteriors; the
/// compiled route that serves it is the accepted-posterior pooling, which calls the same engine.
#[test]
fn the_pooled_exact_posterior_is_the_mixture_of_the_per_table_engines() {
    on_large_stack("twin-pooled", || {
        let params = BayesianDiscoverParams::default();
        let tables: Vec<TabularData> =
            [31_u64, 32].iter().map(|&seed| chain_table(150, seed)).collect();
        let singles: Vec<GraphPosterior> = tables
            .iter()
            .map(|table| {
                posterior(&run(&static_study(
                    table,
                    DiscoverySpec::exact_dag_posterior(params.clone()),
                )))
            })
            .collect();
        let pooled = antecedent::discovery::discover_pooled_exact_dag_posterior(
            &tables,
            &three(),
            &params,
            &ctx(),
        )
        .unwrap();
        let mixture = antecedent::discovery::pool_posteriors_over_imputations(&singles).unwrap();
        same_posterior(&mixture, &pooled);
    });
}

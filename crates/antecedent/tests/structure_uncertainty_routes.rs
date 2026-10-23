//! Uncertainty over a discovered structure, through the compiler.
//!
//! A class posterior collapses the DAG posterior a search lists into a posterior over
//! Markov equivalence classes, and a stability search re-runs a frequentist search on
//! resamples drawn under the declared sampling design and tabulates what it returned. Both
//! compile to a complete lowered program with a stable identity, the plan view reads that
//! program, and the answers equal what the engines return on the same fixtures.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "integration test with deterministic fixtures and bitwise comparisons"
)]

use antecedent::discovery::{
    BayesianDiscoverParams, DiscoverParams, GraphMcmcSchedule, StaticDiscoverParams,
};
use antecedent::discovery_defaults::resolve_ci;
use antecedent::planner::lowered::{LoweringDepth, UncertaintyReading};
use antecedent::planner::statistical::{ResampleUnit, ResamplingMethod};
use antecedent::{CompiledExecution, DiscoveredStructure, DiscoverySpec, Study};
use antecedent_core::{CausalRng, DeclaredSamplingDesign, ExecutionContext, NodeRef, VariableId};
use antecedent_data::{
    MultiEnvironmentData, SamplingRegularity, TabularData, TimeIndex, TimeSeriesData,
};
use antecedent_discovery::selective::{
    DiscoveredStructure as Canonical, StructureClass, StructureNode,
};
use antecedent_discovery::{BackgroundKnowledge, GraphPosteriorAtomKind, MultiDatasetConstraints};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x5EED_0031)
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

/// `z -> t -> y`, `z -> y` with uniform noise: every pair is strongly dependent.
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

/// `y[t]` is driven by `x[t-1]`.
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

fn three() -> Vec<VariableId> {
    vec![v(0), v(1), v(2)]
}

/// Compile twice (stable identity), then prepare and estimate (stamped identity).
fn compiled(study: &Study, route: &str) -> CompiledExecution {
    let ctx = ctx();
    let plan = study.compile(&ctx).unwrap();
    let lowered = plan.lowered().expect("compile attaches a lowered program");
    assert_eq!(lowered.route(), route);
    assert_eq!(lowered.depth(), LoweringDepth::Complete);
    let identity = lowered.identity();
    let view = study.plan_view(&ctx).unwrap();
    assert_eq!(view.identity(), identity, "the plan view reads the lowered program");
    assert_eq!(view.route().as_str(), route);
    let again = study.compile(&ctx).unwrap();
    assert_eq!(again.lowered_program_identity(), Some(identity), "stable identity");
    let execution = CompiledExecution::run(study, &ctx).unwrap();
    assert_eq!(execution.result().lowered_program_identity(), Some(identity));
    assert!(execution.result().executed_contract.is_some(), "the estimate stamps its contract");
    execution
}

fn node(raw: u32) -> StructureNode {
    StructureNode::from_node_ref(NodeRef::Static(v(raw)))
}

#[test]
fn a_class_posterior_collapses_the_dag_posterior_it_wraps() {
    on_large_stack("class-posterior", || {
        let data = chain_table(300, 5);
        let dag_study = Study::tabular(data.clone())
            .discovery(
                DiscoverySpec::exact_dag_posterior(BayesianDiscoverParams::default()),
                three(),
            )
            .build()
            .unwrap();
        let class_spec = DiscoverySpec::exact_dag_posterior(BayesianDiscoverParams::default())
            .equivalence_classes();
        let study = Study::tabular(data.clone()).discovery(class_spec, three()).build().unwrap();
        assert_eq!(&*study.inspect().unwrap().inference, "bayesian");
        let execution = compiled(&study, "DiscoverStatic");
        let lowered = execution.prepared().plan().lowered().unwrap();
        assert_eq!(lowered.claim().uncertainty(), UncertaintyReading::Posterior);
        assert_ne!(
            lowered.identity(),
            dag_study.compile(&ctx()).unwrap().lowered_program_identity().unwrap(),
            "a class posterior is another program than the DAG posterior it collapses"
        );

        let Some(DiscoveredStructure::Posterior(classes)) =
            execution.result().discovery().map(|payload| payload.structure.clone())
        else {
            panic!("a class posterior returns a posterior");
        };
        let direct = antecedent::discovery::discover_exact_dag_posterior(
            &data,
            &three(),
            &BayesianDiscoverParams::default(),
            &ctx(),
        )
        .unwrap();
        let expected = direct.to_equivalence_classes().unwrap();
        assert_eq!(classes.atom_kind, GraphPosteriorAtomKind::Cpdag);
        assert_eq!(classes.atoms, expected.atoms);
        let bits = |weights: &[f64]| weights.iter().map(|w| w.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&classes.weights[..]), bits(&expected.weights[..]));
        // Three variables have 25 DAGs in 11 classes; the collapse never splits a class.
        assert!(classes.n_graphs <= 11);
        assert!(classes.n_graphs < direct.n_graphs);
        let mass: f64 = classes.weights.iter().sum();
        assert!((mass - 1.0).abs() < 1e-9);
        assert_eq!(classes.algorithm.as_deref(), Some("class_posterior(exact_dag_posterior)"));

        // The accepted class posterior is a posterior, never one DAG.
        let (_, claim) = execution.contract_and_claim(&ctx()).unwrap();
        let accepted = antecedent::accept_discovered_posterior(execution.result(), &claim).unwrap();
        assert!(accepted.map_dag().is_err());

        // A search that lists no static graphs is refused when the study compiles.
        let refused = Study::tabular(data)
            .discovery(DiscoverySpec::pc(static_params(), "parcorr").equivalence_classes(), three())
            .build()
            .and_then(|study| study.compile(&ctx()));
        assert!(refused.is_err());
    });
}

#[test]
fn a_stability_search_tabulates_the_resamples_of_its_engine() {
    on_large_stack("stability-discovery", || {
        let data = chain_table(300, 3);
        let spec = DiscoverySpec::pc(static_params(), "parcorr").bootstrap_stability(20);
        let study = Study::tabular(data.clone()).discovery(spec.clone(), three()).build().unwrap();
        assert_eq!(&*study.inspect().unwrap().inference, "frequentist");
        let execution = compiled(&study, "DiscoverStatic");
        let lowered = execution.prepared().plan().lowered().unwrap();
        assert_eq!(lowered.claim().uncertainty(), UncertaintyReading::Sampling);
        let resampling = lowered.statistical().resampling().expect("a stability search resamples");
        assert_eq!(resampling.replicates(), 20);
        assert_eq!(resampling.method(), ResamplingMethod::Bootstrap);
        assert_eq!(resampling.unit(), ResampleUnit::Row);

        let Some(DiscoveredStructure::Stability(table)) =
            execution.result().discovery().map(|payload| payload.structure.clone())
        else {
            panic!("a stability search returns a stability table");
        };
        assert_eq!(table.requested, 20);
        assert_eq!(table.succeeded + table.failed, 20);
        assert!(table.succeeded >= 10);
        // The reference is the structure the engine returns on the full sample.
        let direct =
            antecedent::discovery::discover_pc(&data, &three(), &static_params(), &ctx()).unwrap();
        assert_eq!(table.reference, Canonical::from_cpdag(&direct.evidence.graph));
        // Every pair is strongly dependent, so the strong pairs are adjacent in every resample.
        assert!(table.adjacency_frequency(node(0), node(1)) >= 0.9);
        assert!(table.adjacency_frequency(node(1), node(2)) >= 0.9);
        for edge in &table.edges {
            assert!((0.0..=1.0).contains(&edge.frequency));
            let marks: f64 = edge.marks.iter().map(|mark| mark.frequency).sum();
            assert!((marks - edge.frequency).abs() < 1e-9, "marks split the adjacency frequency");
        }
        let mass: f64 = table.structures.iter().map(|s| s.frequency).sum();
        assert!((mass - 1.0).abs() < 1e-9);

        // Resamples come from keyed streams: the same context returns the same table.
        let again = CompiledExecution::run(&study, &ctx()).unwrap();
        let Some(DiscoveredStructure::Stability(second)) =
            again.result().discovery().map(|payload| payload.structure.clone())
        else {
            panic!("a stability search returns a stability table");
        };
        assert_eq!(second, table);

        // The replicate count is part of the program.
        let more = Study::tabular(data.clone())
            .discovery(
                DiscoverySpec::pc(static_params(), "parcorr").bootstrap_stability(21),
                three(),
            )
            .build()
            .unwrap();
        assert_ne!(
            more.compile(&ctx()).unwrap().lowered_program_identity(),
            study.compile(&ctx()).unwrap().lowered_program_identity()
        );

        // A stability table is not a structure: it has no review gate.
        let (_, claim) = execution.contract_and_claim(&ctx()).unwrap();
        assert!(antecedent::accept_discovered(execution.result(), &claim, true).is_err());

        // A design that does not sample independent rows refuses the row resample.
        let refused = Study::tabular(data)
            .discovery(spec, three())
            .sampling_design(DeclaredSamplingDesign::DependentTimeSeries)
            .build()
            .and_then(|study| CompiledExecution::run(&study, &ctx()).map(|_| ()));
        assert!(refused.is_err());

        // Only the supported searches are re-run.
        let posterior = Study::tabular(chain_table(60, 1))
            .discovery(
                DiscoverySpec::exact_dag_posterior(BayesianDiscoverParams::default())
                    .bootstrap_stability(5),
                three(),
            )
            .build()
            .and_then(|study| study.compile(&ctx()));
        assert!(posterior.is_err());
    });
}

#[test]
fn a_series_stability_search_resamples_circular_blocks() {
    on_large_stack("series-stability", || {
        let series = lag1_series(240, 11);
        let vars = vec![v(0), v(1)];
        let study = Study::series(series.clone())
            .discovery(
                DiscoverySpec::pcmci(temporal_params(1), "parcorr").bootstrap_stability(12),
                vars.clone(),
            )
            .build()
            .unwrap();
        let execution = compiled(&study, "DiscoverTemporal");
        let lowered = execution.prepared().plan().lowered().unwrap();
        let resampling = lowered.statistical().resampling().expect("a stability search resamples");
        assert_eq!(resampling.method(), ResamplingMethod::BlockBootstrap);
        assert_eq!(resampling.unit(), ResampleUnit::CalendarBlock);
        let Some(DiscoveredStructure::Stability(table)) =
            execution.result().discovery().map(|payload| payload.structure.clone())
        else {
            panic!("a stability search returns a stability table");
        };
        assert_eq!(table.requested, 12);
        assert!(table.succeeded >= 6);
        let direct =
            antecedent::discovery::discover_pcmci(&series, &vars, &temporal_params(1), &ctx())
                .unwrap();
        assert_eq!(table.reference, Canonical::from_temporal_dag(&direct.evidence.graph));
        // The lagged link x[t-1] -> y[t] is the dependence the series carries.
        assert!(
            table.edges.iter().any(|edge| edge.frequency >= 0.8 && edge.in_reference),
            "the lagged link survives most block resamples"
        );

        // An independent-subject design refuses block resampling of one series.
        let refused = Study::series(series)
            .discovery(
                DiscoverySpec::pcmci(temporal_params(1), "parcorr").bootstrap_stability(12),
                vars,
            )
            .sampling_design(DeclaredSamplingDesign::IndependentSubjectTrajectories)
            .build()
            .and_then(|study| CompiledExecution::run(&study, &ctx()).map(|_| ()));
        assert!(refused.is_err());
    });
}

#[test]
fn a_panel_stability_search_resamples_whole_units() {
    on_large_stack("panel-stability", || {
        let units = vec![lag1_series(100, 12), lag1_series(100, 13), lag1_series(100, 14)];
        let vars = vec![v(0), v(1)];
        let panel = antecedent::panel_of_units(units).unwrap();
        let study = Study::panel(panel)
            .discovery(
                DiscoverySpec::pcmci(temporal_params(1), "parcorr").bootstrap_stability(10),
                vars,
            )
            .build()
            .unwrap();
        let execution = compiled(&study, "DiscoverTemporal");
        let lowered = execution.prepared().plan().lowered().unwrap();
        let resampling = lowered.statistical().resampling().expect("a stability search resamples");
        assert_eq!(resampling.unit(), ResampleUnit::Subject);
        let Some(DiscoveredStructure::Stability(table)) =
            execution.result().discovery().map(|payload| payload.structure.clone())
        else {
            panic!("a stability search returns a stability table");
        };
        assert_eq!(table.requested, 10);
        assert!(table.succeeded >= 5);
    });
}

#[test]
fn latent_variable_and_multi_environment_searches_are_resampled_too() {
    on_large_stack("lpcmci-jpcmci-stability", || {
        let vars = vec![v(0), v(1)];
        let latent = Study::series(lag1_series(200, 15))
            .discovery(
                DiscoverySpec::lpcmci(temporal_params(1), "parcorr").bootstrap_stability(8),
                vars.clone(),
            )
            .build()
            .unwrap();
        let execution = compiled(&latent, "DiscoverTemporal");
        let Some(DiscoveredStructure::Stability(table)) =
            execution.result().discovery().map(|payload| payload.structure.clone())
        else {
            panic!("a stability search returns a stability table");
        };
        assert_eq!(table.class, StructureClass::TemporalPag);
        assert_eq!(table.requested, 8);
        assert!(table.succeeded >= 4);

        let environments =
            MultiEnvironmentData::try_new(vec![lag1_series(120, 21), lag1_series(120, 22)])
                .unwrap();
        let pooled = Study::series_multi(environments)
            .discovery(
                DiscoverySpec::jpcmci_plus(temporal_params(1), "parcorr").bootstrap_stability(8),
                vars,
            )
            .build()
            .unwrap();
        let execution = compiled(&pooled, "DiscoverTemporal");
        let Some(DiscoveredStructure::Stability(table)) =
            execution.result().discovery().map(|payload| payload.structure.clone())
        else {
            panic!("a stability search returns a stability table");
        };
        assert_eq!(table.class, StructureClass::TemporalCpdag);
        assert_eq!(table.requested, 8);
        assert!(table.succeeded >= 4);
    });
}

#[test]
fn a_template_posterior_collapses_onto_temporal_equivalence_classes() {
    on_large_stack("dbn-class-posterior", || {
        let series = lag1_series(200, 17);
        let vars = vec![v(0), v(1)];
        let schedule = GraphMcmcSchedule { n_chains: 2, n_warmup: 50, n_draws: 100, thin: 1 };
        let dbn =
            DiscoverySpec::dbn_posterior(BayesianDiscoverParams::default(), 1, false, schedule);
        let study = Study::series(series.clone())
            .discovery(dbn.clone().equivalence_classes(), vars.clone())
            .build()
            .unwrap();
        let execution = compiled(&study, "DiscoverTemporal");
        let Some(DiscoveredStructure::Posterior(classes)) =
            execution.result().discovery().map(|payload| payload.structure.clone())
        else {
            panic!("a class posterior returns a posterior");
        };
        assert_eq!(classes.atom_kind, GraphPosteriorAtomKind::Cpdag);
        assert_eq!(classes.max_lag, Some(1));
        assert!(classes.lag_masks.is_some());
        let mass: f64 = classes.weights.iter().sum();
        assert!((mass - 1.0).abs() < 1e-9);
        let templates = antecedent::discovery::discover_dbn_posterior(
            &series,
            &vars,
            &BayesianDiscoverParams::default(),
            1,
            false,
            &schedule,
            &ctx(),
        )
        .unwrap();
        assert!(classes.n_graphs <= templates.n_graphs);
    });
}

#[test]
fn an_order_posterior_under_laplace_noise_orients_what_the_gaussian_posterior_cannot() {
    on_large_stack("laplace-order-posterior", || {
        let n = 1500;
        let mut rng = CausalRng::from_seed(77);
        let mut laplace = || {
            let u = rng.next_f64() - 0.5;
            -u.signum() * (1.0 - 2.0 * u.abs()).max(1e-12).ln()
        };
        let x: Vec<f64> = (0..n).map(|_| laplace()).collect();
        let y: Vec<f64> = x.iter().map(|x| 1.5 * x + laplace()).collect();
        let data =
            TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
        let schedule = GraphMcmcSchedule { n_chains: 4, n_warmup: 200, n_draws: 400, thin: 1 };
        let mut params = BayesianDiscoverParams::default();
        params.score_family = antecedent_state::GraphScoreFamily::LaplaceBic;
        let study = Study::tabular(data)
            .discovery(DiscoverySpec::order_mcmc(params, schedule, false), vec![v(0), v(1)])
            .build()
            .unwrap();
        let execution = compiled(&study, "DiscoverStatic");
        let Some(DiscoveredStructure::Posterior(posterior)) =
            execution.result().discovery().map(|payload| payload.structure.clone())
        else {
            panic!("an order search returns a posterior");
        };
        let forward = posterior.edge_marginals[1];
        let backward = posterior.edge_marginals[2];
        assert!(forward > backward + 0.3, "x -> y {forward}, y -> x {backward}");
    });
}

#[test]
fn a_pag_posterior_is_a_compiled_bayesian_route_over_ancestral_graph_classes() {
    on_large_stack("pag-posterior", || {
        let data = chain_table(1200, 41);
        let study = Study::tabular(data.clone())
            .discovery(
                DiscoverySpec::exact_pag_posterior(BayesianDiscoverParams::default()),
                three(),
            )
            .build()
            .unwrap();
        assert_eq!(&*study.inspect().unwrap().inference, "bayesian");
        let execution = compiled(&study, "DiscoverStatic");
        let lowered = execution.prepared().plan().lowered().unwrap();
        assert_eq!(lowered.claim().uncertainty(), UncertaintyReading::Posterior);
        let Some(DiscoveredStructure::Posterior(pag)) =
            execution.result().discovery().map(|payload| payload.structure.clone())
        else {
            panic!("a PAG posterior returns a posterior");
        };
        assert_eq!(pag.atom_kind, GraphPosteriorAtomKind::Pag);
        assert!(pag.mark_atoms.is_some());
        let mass: f64 = pag.weights.iter().sum();
        assert!((mass - 1.0).abs() < 1e-9);
        let direct =
            antecedent::discovery::discover_exact_pag_posterior(&data, &three(), &ctx()).unwrap();
        assert_eq!(pag.atoms, direct.atoms);
        // Five variables are not enumerated.
        let wide = TabularData::from_f64_columns([
            ("a", &[1.0, 2.0, 3.0, 4.0][..]),
            ("b", &[2.0, 1.0, 4.0, 3.0][..]),
            ("c", &[1.0, 3.0, 2.0, 4.0][..]),
            ("d", &[4.0, 1.0, 3.0, 2.0][..]),
            ("e", &[1.0, 4.0, 2.0, 3.0][..]),
        ])
        .unwrap();
        let refused = Study::tabular(wide)
            .discovery(
                DiscoverySpec::exact_pag_posterior(BayesianDiscoverParams::default()),
                vec![v(0), v(1), v(2), v(3), v(4)],
            )
            .build()
            .and_then(|study| CompiledExecution::run(&study, &ctx()).map(|_| ()));
        assert!(refused.is_err());
    });
}

#[test]
fn a_regime_search_is_resampled_with_its_labels_one_table_per_regime() {
    on_large_stack("rpcmci-stability", || {
        let n = 300;
        let series = lag1_series(n, 31);
        let vars = vec![v(0), v(1)];
        let regimes = antecedent::discovery::two_regime_half_split(n);
        let spec = DiscoverySpec::rpcmci(temporal_params(1), regimes, Some(40), "parcorr")
            .bootstrap_stability(8);
        let study = Study::series(series).discovery(spec, vars).build().unwrap();
        let execution = compiled(&study, "DiscoverTemporal");
        let Some(DiscoveredStructure::RegimeStability(tables)) =
            execution.result().discovery().map(|payload| payload.structure.clone())
        else {
            panic!("a regime stability search returns one table per regime");
        };
        assert_eq!(tables.len(), 2);
        for (_, table) in &tables {
            assert_eq!(table.requested, 8);
            assert_eq!(table.class, StructureClass::TemporalCpdag);
        }
    });
}

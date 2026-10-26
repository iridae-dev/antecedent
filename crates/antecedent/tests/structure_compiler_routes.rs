//! Structure discovery, attribution and simulation through the compiler.
//!
//! Each route compiles to a complete lowered program with a stable identity,
//! preparation retains it and an estimate stamps it; the answers equal the
//! search, attribution and simulation engines' own results bit for bit on the
//! same fixtures and contexts; a discovered structure's provenance reaches the
//! contract and the program identity of a study prepared on it; and the hidden
//! engines are not on the public surface.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "integration test with deterministic fixtures and bitwise comparisons"
)]

use std::sync::Arc;

use antecedent::analysis::RetainedOperation;
use antecedent::discovery::{BayesianDiscoverParams, DiscoverParams, StaticDiscoverParams};
use antecedent::discovery_defaults::resolve_ci;
use antecedent::planner::lowered::{LoweringDepth, ProblemTarget};
use antecedent::{
    AttributionOutcome, CompiledExecution, DiscoveredStructure, DiscoverySpec, SimulationOutcome,
    StructureSource, Study, accept_discovered,
};
use antecedent_core::{
    AttributeQuery, AttributeTarget, CausalQuery, CausalRng, ExecutionContext, Intervention,
    MechanismChangeQuery, PopulationSelector, SimulateQuery, SimulationTarget, UnitChangeQuery,
    Value, VariableId,
};
use antecedent_data::{SamplingRegularity, TableView, TabularData, TimeIndex, TimeSeriesData};
use antecedent_discovery::{BackgroundKnowledge, DiscoveryResult, MultiDatasetConstraints};
use antecedent_graph::{Dag, DenseNodeId};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x5EED_0017)
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

/// `z -> t -> y`, `z -> y` with uniform (non-Gaussian) noise, so every search
/// has a structure to find and `LiNGAM` can orient it.
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

fn chain_dag() -> Dag {
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(0, 1), (1, 2), (0, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

/// Two periods of `x -> y`; the `y` mechanism shifts in the second.
fn two_period_table() -> TabularData {
    let x: Vec<f64> = (0..80_u32).map(|i| f64::from(i % 40) * 0.1).collect();
    let y: Vec<f64> = (0..80_u32)
        .map(|i| {
            let x = f64::from(i % 40) * 0.1;
            if i < 40 { 1.0 + 2.0 * x } else { 6.0 + 2.0 * x }
        })
        .collect();
    TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap()
}

fn two_node_dag() -> Dag {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph
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
    assert_eq!(view.statistical().identity(), lowered.statistical().identity());
    assert_eq!(view.evidence(), lowered.evidence());
    let again = study.compile(&ctx).unwrap();
    assert_eq!(again.lowered_program_identity(), Some(identity), "stable identity");
    let execution = CompiledExecution::run(study, &ctx).unwrap();
    assert_eq!(execution.prepared().plan().lowered_program_identity(), Some(identity));
    assert_eq!(execution.result().lowered_program_identity(), Some(identity));
    assert!(execution.result().executed_contract.is_some(), "the estimate stamps its contract");
    execution
}

/// The graph, its scored links and its separating sets. Separating sets live in a
/// hash map, so they are compared as sorted entries rather than in iteration order.
fn graph_and_links<G: std::fmt::Debug, R>(
    result: &DiscoveryResult<G, R>,
) -> (String, String, Vec<String>) {
    let mut sepsets: Vec<String> =
        result.sepsets.iter().map(|entry| format!("{entry:?}")).collect();
    sepsets.sort_unstable();
    (format!("{:?}", result.evidence.graph), format!("{:?}", result.evidence.links), sepsets)
}

#[test]
fn static_searches_compile_and_equal_their_engines() {
    on_large_stack("static-discovery", || {
        let data = chain_table(400, 3);
        // PC.
        let study = Study::tabular(data.clone())
            .discovery(DiscoverySpec::pc(static_params(), "parcorr"), three())
            .build()
            .unwrap();
        let inspected = study.inspect().unwrap();
        assert_eq!(inspected.structure_source, StructureSource::Discovered);
        assert_eq!(inspected.graph_class.as_str(), "Cpdag", "a CPDAG stays a class");
        assert_eq!(&*inspected.query_kind, "Discover");
        assert_eq!(&*inspected.inference, "frequentist");
        let execution = compiled(&study, "DiscoverStatic");
        let Some(DiscoveredStructure::StaticCpdag(compiled_pc)) =
            execution.result().discovery().map(|payload| payload.structure.clone())
        else {
            panic!("PC returns a static CPDAG");
        };
        let direct =
            antecedent::discovery::discover_pc(&data, &three(), &static_params(), &ctx()).unwrap();
        assert_eq!(graph_and_links(&compiled_pc), graph_and_links(&direct));
        let (contract, claim) = execution.contract_and_claim(&ctx()).unwrap();
        assert_eq!(
            execution.result().executed_contract.as_ref().map(|executed| &executed.identities),
            Some(&claim.identities),
            "the claim seals the stamped execution"
        );
        assert_eq!(contract.discovery_algorithm, None, "a search reads no accepted graph");

        // GES and LiNGAM.
        let ges = Study::tabular(data.clone())
            .discovery(DiscoverySpec::ges(static_params(), "parcorr"), three())
            .build()
            .unwrap();
        let ges_execution = compiled(&ges, "DiscoverStatic");
        let Some(DiscoveredStructure::StaticCpdag(compiled_ges)) =
            ges_execution.result().discovery().map(|payload| payload.structure.clone())
        else {
            panic!("GES returns a static CPDAG");
        };
        let direct_ges =
            antecedent::discovery::discover_ges(&data, &three(), &static_params(), &ctx()).unwrap();
        assert_eq!(graph_and_links(&compiled_ges), graph_and_links(&direct_ges));
        let lingam = Study::tabular(data.clone())
            .discovery(DiscoverySpec::lingam(static_params(), 0.1, "parcorr"), three())
            .build()
            .unwrap();
        let lingam_execution = compiled(&lingam, "DiscoverStatic");
        let Some(DiscoveredStructure::StaticDag(compiled_lingam)) =
            lingam_execution.result().discovery().map(|payload| payload.structure.clone())
        else {
            panic!("LiNGAM returns a static DAG");
        };
        let direct_lingam =
            antecedent::discovery::discover_lingam(&data, &three(), &static_params(), 0.1, &ctx())
                .unwrap();
        assert_eq!(graph_and_links(&compiled_lingam), graph_and_links(&direct_lingam));

        // Two searches that differ only in a bound are different programs.
        let mut loose = static_params();
        loose.alpha = 0.2;
        let other = Study::tabular(data)
            .discovery(DiscoverySpec::pc(loose, "parcorr"), three())
            .build()
            .unwrap();
        assert_ne!(
            other.compile(&ctx()).unwrap().lowered_program_identity(),
            study.compile(&ctx()).unwrap().lowered_program_identity()
        );
    });
}

#[test]
fn a_structure_posterior_declares_bayesian_inference_and_equals_its_engine() {
    on_large_stack("posterior-discovery", || {
        let data = chain_table(300, 5);
        let study = Study::tabular(data.clone())
            .discovery(
                DiscoverySpec::exact_dag_posterior(BayesianDiscoverParams::default()),
                three(),
            )
            .build()
            .unwrap();
        let execution = compiled(&study, "DiscoverStatic");
        let (contract, _) = execution.contract_and_claim(&ctx()).unwrap();
        assert_eq!(&*contract.inference, "bayesian", "the method declares its inference");
        let lowered = execution.prepared().plan().lowered().unwrap();
        assert!(matches!(lowered.scientific().target(), ProblemTarget::QueryFunctional { .. }));
        let Some(DiscoveredStructure::Posterior(posterior)) =
            execution.result().discovery().map(|payload| payload.structure.clone())
        else {
            panic!("the exact search returns a posterior");
        };
        let direct = antecedent::discovery::discover_exact_dag_posterior(
            &data,
            &three(),
            &BayesianDiscoverParams::default(),
            &ctx(),
        )
        .unwrap();
        let bits = |weights: &[f64]| weights.iter().map(|w| w.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&posterior.weights[..]), bits(&direct.weights[..]));
        assert_eq!(posterior.adjacency, direct.adjacency);
        // A static search refuses lagged data and a Bayesian study setting.
        assert!(
            Study::tabular(data.clone())
                .discovery(
                    DiscoverySpec::exact_dag_posterior(BayesianDiscoverParams::default()),
                    three()
                )
                .inference(antecedent::InferenceMode::Bayesian(
                    antecedent::BayesianConfig::conjugate()
                ))
                .build()
                .is_err()
        );
    });
}

#[test]
fn temporal_searches_compile_and_equal_their_engines() {
    on_large_stack("temporal-discovery", || {
        let series = lag1_series(200, 11);
        let vars = vec![v(0), v(1)];
        let study = Study::series(series.clone())
            .discovery(DiscoverySpec::pcmci_plus(temporal_params(1), "parcorr"), vars.clone())
            .build()
            .unwrap();
        let execution = compiled(&study, "DiscoverTemporal");
        let Some(DiscoveredStructure::TemporalCpdag(compiled_plus)) =
            execution.result().discovery().map(|payload| payload.structure.clone())
        else {
            panic!("PCMCI+ returns a temporal CPDAG");
        };
        let direct =
            antecedent::discovery::discover_pcmci_plus(&series, &vars, &temporal_params(1), &ctx())
                .unwrap();
        assert_eq!(graph_and_links(&compiled_plus), graph_and_links(&direct));
        assert_eq!(execution.inspect().unwrap().graph_class.as_str(), "TemporalCpdag");

        // Panel: each unit is its own series.
        let units = vec![lag1_series(120, 12), lag1_series(120, 13)];
        let panel = antecedent::panel_of_units(units.clone()).unwrap();
        let panel_study = Study::panel(panel)
            .discovery(DiscoverySpec::pcmci(temporal_params(1), "parcorr"), vars.clone())
            .build()
            .unwrap();
        let panel_execution = compiled(&panel_study, "DiscoverTemporal");
        let Some(DiscoveredStructure::TemporalDag(compiled_panel)) =
            panel_execution.result().discovery().map(|payload| payload.structure.clone())
        else {
            panic!("PCMCI returns a temporal DAG");
        };
        let direct_panel =
            antecedent::discovery::discover_pcmci_panel(&units, &vars, &temporal_params(1), &ctx())
                .unwrap();
        assert_eq!(graph_and_links(&compiled_panel), graph_and_links(&direct_panel));

        // A temporal search refuses tabular data.
        let refused = Study::tabular(chain_table(50, 1))
            .discovery(DiscoverySpec::pcmci(temporal_params(1), "parcorr"), three())
            .build()
            .and_then(|study| study.compile(&ctx()));
        assert!(refused.is_err());
    });
}

#[test]
fn discovered_provenance_reaches_the_downstream_contract_and_program() {
    on_large_stack("discovered-provenance", || {
        let data = chain_table(400, 4);
        let search = Study::tabular(data.clone())
            .discovery(DiscoverySpec::lingam(static_params(), 0.1, "parcorr"), three())
            .build()
            .unwrap();
        let execution = CompiledExecution::run(&search, &ctx()).unwrap();
        let (_, claim) = execution.contract_and_claim(&ctx()).unwrap();
        let accepted = accept_discovered(execution.result(), &claim, true).unwrap();
        let provenance = accepted.provenance().expect("the review gate records provenance").clone();
        assert_eq!(&*provenance.method, "direct_lingam");
        assert_eq!(&*provenance.inference, "frequentist");
        assert_eq!(&*provenance.discovered_class, "dag");
        assert_eq!(provenance.claim, *claim.claim_id.as_bytes());
        assert_eq!(Some(provenance.program), execution.lowered_identity().map(|id| *id.as_bytes()));
        let dag = accepted.as_dag().expect("LiNGAM's review accepts a DAG").clone();
        let query = CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::Paths {
            sources: Arc::from([v(1)]),
            outcome: v(2),
            max_paths: 16,
            max_len: 4,
            through: Arc::from([]),
        }));
        let on_discovered =
            Study::tabular(data.clone()).graph(accepted).query(query.clone()).build().unwrap();
        let on_supplied = Study::tabular(data).graph(dag).query(query).build().unwrap();
        let discovered = CompiledExecution::run(&on_discovered, &ctx()).unwrap();
        let supplied = CompiledExecution::run(&on_supplied, &ctx()).unwrap();
        let (downstream, _) = discovered.contract_and_claim(&ctx()).unwrap();
        let (plain, _) = supplied.contract_and_claim(&ctx()).unwrap();
        assert_eq!(downstream.structure_source, StructureSource::Accepted);
        assert_eq!(downstream.structure_provenance.as_deref(), Some(&provenance));
        assert_eq!(downstream.discovery_algorithm.as_deref(), Some("direct_lingam"));
        assert!(plain.structure_provenance.is_none());
        // Identification excludes the discovery audit data; the program binds it.
        assert_eq!(downstream.identities.identification, plain.identities.identification);
        assert_ne!(downstream.identities.program, plain.identities.program);
        assert_ne!(discovered.lowered_identity(), supplied.lowered_identity());
        // The fit itself is the same: the answer does not depend on where the graph came from.
        let total = |execution: &CompiledExecution| match &execution
            .result()
            .attribution()
            .expect("attribution payload")
            .outcome
        {
            AttributionOutcome::Change(change) => change.total_change.to_bits(),
            other => panic!("paths return a change attribution, got {other:?}"),
        };
        assert_eq!(total(&discovered), total(&supplied));
    });
}

#[test]
fn attribution_and_simulation_equal_their_engines() {
    on_large_stack("model-routes", || {
        let data = chain_table(300, 9);
        let fitted = antecedent::gcm::fit_gcm(chain_dag(), &data).unwrap();
        let run = |query: CausalQuery, route: &str| {
            let study =
                Study::tabular(data.clone()).graph(chain_dag()).query(query).build().unwrap();
            compiled(&study, route)
        };
        // Paths.
        let paths = run(
            CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::Paths {
                sources: Arc::from([v(0)]),
                outcome: v(2),
                max_paths: 16,
                max_len: 4,
                through: Arc::from([]),
            })),
            "GcmAttributeDag",
        );
        let direct =
            antecedent::gcm::attribute_paths(&fitted.model, &[v(0)], v(2), 16, 4, &ctx()).unwrap();
        let Some(AttributionOutcome::Change(change)) =
            paths.result().attribution().map(|payload| payload.outcome.clone())
        else {
            panic!("paths return a change attribution");
        };
        assert_eq!(change.total_change.to_bits(), direct.total_change.to_bits());
        assert_eq!(format!("{:?}", change.path_breakdown), format!("{:?}", direct.path_breakdown));
        assert_eq!(
            paths.result().attribution().unwrap().mechanisms.len(),
            fitted.assignments.len(),
            "the selection record of every node is published"
        );
        let (contract, _) = paths.contract_and_claim(&ctx()).unwrap();
        assert_eq!(&*contract.query_kind, "Attribute");
        // Interventional sample.
        let interventions: Arc<[Intervention]> =
            Arc::from([Intervention::set(v(1), Value::f64(1.0))]);
        let sample = run(
            CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::InterventionalSample {
                interventions: Arc::clone(&interventions),
                draws: 64,
                seed: 4,
            })),
            "GcmSimulateDag",
        );
        let mut rng = CausalRng::from_seed(4);
        let direct_sample =
            antecedent::gcm::sample_do(&fitted.model, &interventions, 64, &mut rng, &ctx())
                .unwrap();
        let Some(SimulationOutcome::Sample(batch)) =
            sample.result().simulation().map(|payload| payload.outcome.clone())
        else {
            panic!("a sample returns draws");
        };
        let bits = |values: &[f64]| values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&batch.values[..]), bits(&direct_sample.values[..]));
        // Policy value and unit effects.
        let value = run(
            CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::PolicyValue {
                interventions: Arc::clone(&interventions),
                outcome: v(2),
                draws: 64,
            })),
            "GcmSimulateDag",
        );
        let direct_value =
            antecedent::gcm::policy_value(&fitted.model, &interventions, v(2), 64, &ctx()).unwrap();
        let Some(SimulationOutcome::PolicyValue(compiled_value)) =
            value.result().simulation().map(|payload| payload.outcome.clone())
        else {
            panic!("a policy value returns its value");
        };
        assert_eq!(compiled_value, direct_value);
        let ite = run(
            CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::UnitEffects {
                treatment: v(1),
                outcome: v(2),
                active: 1.0,
                control: 0.0,
            })),
            "GcmSimulateDag",
        );
        let direct_ite = antecedent::gcm::counterfactual_ite(
            fitted.model.clone(),
            &data,
            v(1),
            v(2),
            1.0,
            0.0,
            &ctx(),
        )
        .unwrap();
        let Some(SimulationOutcome::UnitEffects(compiled_ite)) =
            ite.result().simulation().map(|payload| payload.outcome.clone())
        else {
            panic!("unit effects return the effects");
        };
        assert_eq!(bits(&compiled_ite.unit_effects[..]), bits(&direct_ite.unit_effects[..]));
        assert_eq!(compiled_ite.mean_ite.to_bits(), direct_ite.mean_ite.to_bits());
    });
}

#[test]
fn mechanism_and_unit_change_run_on_the_retained_path() {
    on_large_stack("retained-change", || {
        let data = two_period_table();
        let fitted = antecedent::gcm::fit_gcm(two_node_dag(), &data).unwrap();
        let detection = MechanismChangeQuery::new(
            [v(0), v(1)],
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
            0.05,
            10,
        );
        let study = Study::tabular(data.clone())
            .graph(two_node_dag())
            .query(CausalQuery::MechanismChange(detection.clone()))
            .build()
            .unwrap();
        let execution = compiled(&study, "GcmDag");
        assert!(
            matches!(execution.prepared().retained_operation(), RetainedOperation::Attribution(_)),
            "mechanism-change detection is sealed on the retained path"
        );
        let direct = antecedent::gcm::mechanism_change_detection(
            &fitted.model,
            &data,
            &detection,
            antecedent::gcm::MechanismChangeMethod::LikelihoodRatio,
            &ctx(),
        )
        .unwrap();
        let compiled_detections = execution.result().mechanism_change().unwrap();
        assert_eq!(format!("{compiled_detections:?}"), format!("{direct:?}"));
        let unit = UnitChangeQuery::new(v(1), data.row_count());
        let unit_study = Study::tabular(data.clone())
            .graph(two_node_dag())
            .query(CausalQuery::UnitChange(unit.clone()))
            .build()
            .unwrap();
        let unit_execution = compiled(&unit_study, "GcmDag");
        assert!(matches!(
            execution.prepared().retained_operation(),
            RetainedOperation::Attribution(_)
        ));
        let direct_unit =
            antecedent::gcm::attribute_unit_change(&fitted.model, &data, &unit, &ctx()).unwrap();
        assert_eq!(
            format!(
                "{:?}",
                unit_execution.result().payload.unit_change().unwrap().mean_contributions
            ),
            format!("{:?}", direct_unit.mean_contributions)
        );
    });
}

#[test]
fn a_cancelled_or_unleased_route_refuses_before_allocating() {
    on_large_stack("route-resources", || {
        let data = chain_table(200, 2);
        let study = Study::tabular(data)
            .discovery(DiscoverySpec::pc(static_params(), "parcorr"), three())
            .build()
            .unwrap();
        let prepared = study.prepare(&ctx()).unwrap();
        let cancelled = ctx();
        cancelled.cancellation.cancel();
        assert!(matches!(
            prepared.estimate_retained(&cancelled),
            Err(antecedent::CausalError::Cancelled { .. })
        ));
        let tight = ctx().with_admission(antecedent_core::execution::AdmissionPool::fixed(1, 64));
        assert!(matches!(
            prepared.estimate_retained(&tight),
            Err(antecedent::CausalError::Resource { .. })
        ));
    });
}

/// The hidden engines are off the public surface, and the bindings run the compiled routes.
#[test]
fn the_demoted_engines_are_not_on_the_public_surface() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let snapshot = std::fs::read_to_string(root.join("parity/public_api/antecedent.txt")).unwrap();
    let demoted = [
        "antecedent::discovery::discover_",
        "antecedent::gcm::sample_do(",
        "antecedent::gcm::policy_value(",
        "antecedent::gcm::counterfactual_ite(",
        "antecedent::gcm::anomaly_attribution",
        "antecedent::gcm::attribute_",
        "antecedent::gcm::mechanism_change_detection(",
        "antecedent::gcm::sample_interventional_distribution(",
        "antecedent::gcm::rank_root_causes(",
        "antecedent::gcm::sample_with_noise(",
        "antecedent::gcm::noise_table_from_columns(",
        "antecedent::gcm::replay_noise(",
        "antecedent::gcm::counterfactual_policy(",
        "antecedent::gcm::sample_temporal_policy(",
        "antecedent::gcm::counterfactual_temporal_policy(",
    ];
    for line in snapshot.lines().filter(|line| line.starts_with("pub fn ")) {
        for name in demoted {
            assert!(!line.contains(name), "a demoted engine is public: {line}");
        }
    }
    for binding in ["python/src/discovery_api.rs", "python/src/bayesian.rs"] {
        let text = std::fs::read_to_string(root.join(binding)).unwrap();
        assert!(
            !text.contains("discovery::discover_"),
            "{binding} calls a hidden discovery engine instead of the compiled route"
        );
    }
    for (binding, engines) in [
        ("python/src/gcm_api.rs", &["facade_replay_noise", "facade_sample_with_noise"][..]),
        (
            "python/src/intervention_program_api.rs",
            &[
                "facade_counterfactual_policy",
                "facade_sample_temporal_policy",
                "facade_counterfactual_temporal_policy",
            ][..],
        ),
        ("python/src/lib.rs", &["facade_sample_do"][..]),
    ] {
        let text = std::fs::read_to_string(root.join(binding)).unwrap();
        for engine in engines {
            assert!(!text.contains(engine), "{binding} calls the hidden engine {engine}");
        }
    }
}

#[test]
fn an_accepted_structure_posterior_carries_its_provenance_downstream() {
    on_large_stack("posterior-provenance", || {
        let data = chain_table(300, 6);
        let search = Study::tabular(data.clone())
            .discovery(
                DiscoverySpec::exact_dag_posterior(BayesianDiscoverParams::default()),
                three(),
            )
            .build()
            .unwrap();
        let execution = CompiledExecution::run(&search, &ctx()).unwrap();
        let accepted = execution.accept_posterior(&ctx()).unwrap();
        assert_eq!(&*accepted.provenance().method, "exact_dag_posterior");
        assert_eq!(&*accepted.provenance().inference, "bayesian");
        assert_eq!(&*accepted.provenance().review, "posterior");
        assert_eq!(
            Some(accepted.provenance().program),
            execution.lowered_identity().map(|id| *id.as_bytes())
        );
        let query = antecedent::AverageEffectQuery::binary_ate(v(1), v(2));
        let bayesian =
            || antecedent::InferenceMode::Bayesian(antecedent::BayesianConfig::conjugate());
        let on_discovered = Study::tabular(data.clone())
            .accepted_graph_posterior(accepted.clone())
            .query(query.clone())
            .inference(bayesian())
            .refute(antecedent::RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let on_supplied = Study::tabular(data.clone())
            .graph_posterior(accepted.posterior().clone())
            .query(query)
            .inference(bayesian())
            .refute(antecedent::RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let discovered = on_discovered.inspect().unwrap();
        let supplied = on_supplied.inspect().unwrap();
        assert_eq!(discovered.structure_source, StructureSource::GraphPosterior);
        assert_eq!(discovered.structure_provenance.as_deref(), Some(accepted.provenance()));
        assert!(supplied.structure_provenance.is_none());
        assert_eq!(discovered.identities.identification, supplied.identities.identification);
        // The lowered program reads the provenance as evidence; the supplied posterior adds none.
        assert_ne!(
            on_discovered.compile(&ctx()).unwrap().lowered_program_identity(),
            on_supplied.compile(&ctx()).unwrap().lowered_program_identity()
        );

        // The most probable atom as a single accepted DAG.
        let map = accepted.map_dag().unwrap();
        let map_provenance = map.provenance().expect("the MAP graph keeps the provenance");
        assert_eq!(&*map_provenance.review, "posterior_map");
        assert_eq!(map_provenance.claim, accepted.provenance().claim);
        let best = accepted.posterior().weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(best > 0.0);
        // A frequentist search has no posterior to accept.
        let pc = Study::tabular(data)
            .discovery(DiscoverySpec::pc(static_params(), "parcorr"), three())
            .build()
            .unwrap();
        assert!(CompiledExecution::run(&pc, &ctx()).unwrap().accept_posterior(&ctx()).is_err());
    });
}

#[test]
fn noise_replay_counterfactual_and_rollout_equal_their_engines() {
    on_large_stack("model-programs", || {
        let data = chain_table(200, 12);
        let fitted = antecedent::gcm::fit_gcm(chain_dag(), &data).unwrap();
        let run = |target: SimulationTarget| {
            let study = Study::tabular(data.clone())
                .graph(chain_dag())
                .query(CausalQuery::Simulate(SimulateQuery::new(target)))
                .build()
                .unwrap();
            compiled(&study, "GcmSimulateDag")
        };
        let bits = |values: &[f64]| values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let interventions: Arc<[Intervention]> =
            Arc::from([Intervention::set(v(1), Value::f64(0.5))]);

        // Replay of retained noise.
        let (_, table) =
            antecedent::gcm::sample_with_noise(&fitted.model, &interventions, 16, &ctx()).unwrap();
        let noise: Arc<[f64]> = Arc::from(table.values());
        let replay = run(SimulationTarget::NoiseReplay {
            interventions: Arc::clone(&interventions),
            noise: Arc::clone(&noise),
            n_rows: 16,
            policy_salt: 3,
        });
        let direct_table =
            antecedent::gcm::noise_table_from_columns(&fitted.model, 16, &noise, &ctx()).unwrap();
        let direct_replay =
            antecedent::gcm::replay_noise(&fitted.model, &interventions, &direct_table, 3, &ctx())
                .unwrap();
        let Some(SimulationOutcome::Sample(replayed)) =
            replay.result().simulation().map(|payload| payload.outcome.clone())
        else {
            panic!("a replay returns draws");
        };
        assert_eq!(bits(&replayed.values[..]), bits(&direct_replay.values[..]));

        // Counterfactual values of the factual rows.
        let counterfactual = run(SimulationTarget::Counterfactual {
            interventions: Arc::clone(&interventions),
            outcomes: Arc::from([v(2)]),
        });
        let direct_counterfactual = antecedent::gcm::counterfactual_policy(
            fitted.model.clone(),
            &data,
            &interventions,
            &[v(2)],
            &ctx(),
        )
        .unwrap();
        let Some(SimulationOutcome::Counterfactual(values)) =
            counterfactual.result().simulation().map(|payload| payload.outcome.clone())
        else {
            panic!("a counterfactual returns its values");
        };
        assert_eq!(bits(values.values()), bits(direct_counterfactual.values()));

        // A seeded rollout.
        let schedule = antecedent_core::InterventionSequence::new(vec![
            antecedent_core::SequencedIntervention::new(
                Intervention::set(v(0), Value::f64(1.0)),
                antecedent_core::TemporalPolicy::sustained(0, 3),
            ),
        ]);
        let rollout = run(SimulationTarget::TemporalRollout {
            schedule: schedule.clone(),
            n_rows: 4,
            n_steps: 3,
            window: 0,
            history: Arc::from([]),
            seed: 5,
        });
        let plan = antecedent::gcm::TemporalPlan {
            sequence: &schedule,
            n_steps: 3,
            n_rows: 4,
            window: 0,
            initial_history: &[],
        };
        let mut rng = CausalRng::from_seed(5);
        let direct_path =
            antecedent::gcm::sample_temporal_policy(&fitted.model, &plan, &mut rng, &ctx())
                .unwrap();
        let Some(SimulationOutcome::Temporal(path)) =
            rollout.result().simulation().map(|payload| payload.outcome.clone())
        else {
            panic!("a rollout returns its path");
        };
        assert_eq!(path.steps.len(), direct_path.steps.len());
        for (compiled_step, direct_step) in path.steps.iter().zip(&direct_path.steps) {
            assert_eq!(bits(&compiled_step.values[..]), bits(&direct_step.values[..]));
        }
    });
}

#[test]
fn wrapped_mechanisms_name_themselves_in_the_program() {
    on_large_stack("wrapped-program", || {
        let data = chain_table(120, 13);
        let fitted = Arc::new(antecedent::gcm::fit_gcm(chain_dag(), &data).unwrap());
        let standard = antecedent::MechanismProgram::standard();
        let wrapped = antecedent::MechanismProgram::standard().with_wrapped_prefit(
            Arc::clone(&fitted),
            &data,
            &["y"],
        );
        assert!(wrapped.label().starts_with("gcm.registry.wrapped."));
        assert_ne!(wrapped.label(), standard.label());
        let target = SimulationTarget::InterventionalSample {
            interventions: Arc::from([Intervention::set(v(1), Value::f64(1.0))]),
            draws: 8,
            seed: 1,
        };
        let build = |program: antecedent::MechanismProgram, rows: TabularData| {
            Study::tabular(rows)
                .graph(chain_dag())
                .query(CausalQuery::Simulate(SimulateQuery::new(target.clone())))
                .mechanisms(program)
                .build()
                .unwrap()
        };
        let on_wrapped = build(wrapped.clone(), data.clone());
        let on_standard = build(standard, data);
        assert_ne!(
            on_wrapped.compile(&ctx()).unwrap().lowered_program_identity(),
            on_standard.compile(&ctx()).unwrap().lowered_program_identity()
        );
        // A wrapped model never refits on other rows.
        let other = build(wrapped, chain_table(120, 14));
        assert!(CompiledExecution::run(&other, &ctx()).is_err());
    });
}

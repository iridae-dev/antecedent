//! Per-population discovery through the facade: one declared contract run on each population's
//! rows, nothing pooled, with an edge-by-edge comparison.
//!
//! Known graphs (linear-Gaussian, large sample, strict threshold so that the skeleton is the
//! true one):
//!
//! * `north`: the chain `x -> y -> z`. It has no v-structure, so its class keeps both edges
//!   undirected.
//! * `south`: the collider `x -> y <- z`. The unshielded collider compels both edges.
//!
//! The oracle for the comparison is that hand derivation: both edges are present in both
//! populations and both are marked differently.

use antecedent::analysis::per_population_discovery::{
    PerPopulationDiscovery, PopulationRows, add_per_population_claims, discover_per_population,
    request_selection_targets_from_differences,
};
use antecedent::analysis::selective_discovery::revalidate_after_discovery;
use antecedent_contract::composition_verification::{CompositionGraph, Revalidation};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::selective::{CiTestSpec, Mark, StaticAlgorithm, StaticDiscoveryContract};

const SEED: u64 = 31_337;
const ROWS: usize = 4_000;

struct Lcg(u64);

impl Lcg {
    fn normal(&mut self) -> f64 {
        let mut unit = || {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (f64::from(u32::try_from(self.0 >> 40).unwrap()) + 0.5) / f64::from(1u32 << 24)
        };
        let (u, v) = (unit(), unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn frame(x: &[f64], y: &[f64], z: &[f64]) -> TabularData {
    TabularData::from_f64_columns([("x", x), ("y", y), ("z", z)]).unwrap()
}

/// `x -> y -> z`.
fn chain(seed: u64) -> TabularData {
    let mut rng = Lcg(seed);
    let (mut x, mut y, mut z) = (vec![], vec![], vec![]);
    for _ in 0..ROWS {
        let a = rng.normal();
        let b = 0.8 * a + 0.6 * rng.normal();
        let c = 0.8 * b + 0.6 * rng.normal();
        x.push(a);
        y.push(b);
        z.push(c);
    }
    frame(&x, &y, &z)
}

/// `x -> y <- z`.
fn collider(seed: u64) -> TabularData {
    let mut rng = Lcg(seed);
    let (mut x, mut y, mut z) = (vec![], vec![], vec![]);
    for _ in 0..ROWS {
        let a = rng.normal();
        let c = rng.normal();
        let b = 0.8 * a + 0.8 * c + 0.5 * rng.normal();
        x.push(a);
        y.push(b);
        z.push(c);
    }
    frame(&x, &y, &z)
}

fn contract() -> StaticDiscoveryContract {
    let mut contract = StaticDiscoveryContract::new(
        StaticAlgorithm::Pc,
        (0..3).map(VariableId::from_raw).collect(),
        CiTestSpec::named("parcorr"),
        SEED,
    );
    contract.fdr = None;
    contract.constraints.alpha = 0.001;
    contract
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(SEED)
}

fn rows<'a>(population: &'a str, data: &'a TabularData) -> PopulationRows<'a> {
    PopulationRows { population, data }
}

#[test]
fn each_population_is_discovered_on_its_own_rows_and_the_edges_are_lined_up() {
    let (north, south) = (chain(1), collider(2));
    let driver = discover_per_population(
        &contract(),
        &[rows("south", &south), rows("north", &north)],
        &ctx(),
    )
    .unwrap();
    let report = driver.report().unwrap();

    // Label order, whatever the request order.
    let labels: Vec<&str> = report.populations.iter().map(|p| p.population.as_str()).collect();
    assert_eq!(labels, vec!["north", "south"]);
    assert_eq!(report.populations[0].unresolved_edges, 2);
    assert_eq!(report.populations[1].unresolved_edges, 0);
    assert!(report.populations.iter().all(|p| !p.search_incomplete && p.retained_statistics > 0));
    assert_ne!(report.populations[0].structure_identity, report.populations[1].structure_identity);
    assert_ne!(
        report.populations[0].data_identity.digest,
        report.populations[1].data_identity.digest
    );

    // x - y and y - z are in both structures; both are marked differently.
    assert_eq!(report.edges.len(), 2);
    assert_eq!(report.differing_edges().len(), 2);
    let south_marks = |edge: usize| report.edges[edge].marks[1].1;
    assert_eq!(south_marks(0), Some((Mark::Tail, Mark::Arrow)));
    assert_eq!(south_marks(1), Some((Mark::Arrow, Mark::Tail)));
    assert!(report.edges.iter().all(|e| e.marks[0].1 != e.marks[1].1));

    // Each population keeps its own statistic store.
    assert!(driver.adapter("north").unwrap().retained_statistics() > 0);
    assert!(driver.adapter("south").unwrap().retained_statistics() > 0);
    assert!(driver.adapter("east").is_none());
}

#[test]
fn a_changed_population_recomputes_its_own_result_and_leaves_the_others_untouched() {
    let (north, south) = (chain(1), collider(2));
    let mut driver = PerPopulationDiscovery::new();
    let first =
        driver.run(&contract(), &[rows("north", &north), rows("south", &south)], &ctx()).unwrap();
    let south_before = first.populations[1].clone();
    let north_before = first.populations[0].clone();
    let first_identity = first.identity;

    // New rows for north only, from the same graph.
    let north_again = chain(9);
    let second = driver
        .run(&contract(), &[rows("north", &north_again), rows("south", &south)], &ctx())
        .unwrap();
    let (north_after, south_after) = (&second.populations[0], &second.populations[1]);
    assert_ne!(north_after.data_identity.digest, north_before.data_identity.digest);
    assert_ne!(north_after.result_identity, north_before.result_identity);
    // South is served from its own retained store: identical rows, identical identities, and
    // nothing was computed.
    assert_eq!(south_after.data_identity.digest, south_before.data_identity.digest);
    assert_eq!(south_after.result_identity, south_before.result_identity);
    assert!(south_after.report.plan.is_pure_reuse());
    assert_eq!(south_after.report.work.ci_tests_attempted, 0);
    // The set identity moved because one member did.
    assert_ne!(second.identity, first_identity);
}

#[test]
fn pooled_or_duplicated_populations_and_malformed_requests_are_refused_without_work() {
    let (north, south) = (chain(1), collider(2));
    let refused = |contract: &StaticDiscoveryContract, populations: &[PopulationRows<'_>]| {
        let mut driver = PerPopulationDiscovery::new();
        let error = driver.run(contract, populations, &ctx()).unwrap_err();
        assert!(driver.report().is_none(), "a refusal commits nothing");
        assert!(driver.adapter("north").is_none(), "a refusal retains nothing");
        (error.code().unwrap(), error.detail().unwrap())
    };

    // The same rows under two labels are one dataset, not two populations.
    assert_eq!(
        refused(&contract(), &[rows("north", &north), rows("south", &north)]),
        ("invalid_argument", "selective_discovery.pooled_population_data")
    );
    // A pooled declaration (a further population on the contract) is not per-population.
    let mut pooled = contract();
    pooled.additional_populations = vec!["south".to_owned()];
    assert_eq!(
        refused(&pooled, &[rows("north", &north), rows("south", &south)]),
        ("route_not_supported", "selective_discovery.pooled_population_search")
    );
    assert_eq!(
        refused(&contract(), &[rows("north", &north)]),
        ("invalid_argument", "selective_discovery.populations_missing")
    );
    assert_eq!(
        refused(&contract(), &[rows("north", &north), rows("north", &south)]),
        ("invalid_argument", "selective_discovery.population_repeated")
    );
    assert_eq!(
        refused(&contract(), &[rows("north", &north), rows(" ", &south)]),
        ("invalid_argument", "selective_discovery.population_missing")
    );
}

#[test]
fn each_population_is_its_own_claim_and_a_change_of_one_leaves_the_other_valid() {
    let claims = |north: &TabularData, south: &TabularData| {
        let mut driver = PerPopulationDiscovery::new();
        driver.run(&contract(), &[rows("north", north), rows("south", south)], &ctx()).unwrap();
        let mut graph = CompositionGraph::new();
        let keys = add_per_population_claims(&mut graph, "disc", &driver).unwrap();
        (graph, keys)
    };
    let (before, keys) = claims(&chain(1), &collider(2));
    assert_eq!(keys["north"].graph, "disc.north.graph");
    assert_eq!(keys["south"].graph, "disc.south.graph");
    assert_eq!(keys.len(), 2);

    let (unchanged, _) = claims(&chain(1), &collider(2));
    let result = revalidate_after_discovery(&before, &unchanged).unwrap();
    for key in ["disc.north.graph", "disc.south.graph", "disc.north.data", "disc.south.data"] {
        assert_eq!(result.claim(key).unwrap().class, Revalidation::StillValid, "{key}");
    }

    // New rows for north: north's data, statistics and graph recompute; south's stay valid.
    let (changed, _) = claims(&chain(9), &collider(2));
    let result = revalidate_after_discovery(&before, &changed).unwrap();
    for key in ["disc.north.data", "disc.north.statistics", "disc.north.graph"] {
        assert_eq!(result.claim(key).unwrap().class, Revalidation::Recompute, "{key}");
    }
    for key in ["disc.south.data", "disc.south.statistics", "disc.south.graph"] {
        assert_eq!(result.claim(key).unwrap().class, Revalidation::StillValid, "{key}");
    }
}

#[test]
fn differences_between_populations_are_not_selection_targets() {
    let (north, south) = (chain(1), collider(2));
    let driver = discover_per_population(
        &contract(),
        &[rows("north", &north), rows("south", &south)],
        &ctx(),
    )
    .unwrap();
    let refusal = request_selection_targets_from_differences(
        driver.report().unwrap(),
        &[VariableId::from_raw(1)],
    )
    .unwrap_err();
    assert_eq!(refusal.code, "discovered_structure_unresolved");
    assert_eq!(refusal.detail, "selective_discovery.differences_are_not_selection_targets");
    assert!(refusal.remedy.starts_with("2 edge(s) differ"));
    assert!(refusal.remedy.contains("Declare the selection targets"));
}

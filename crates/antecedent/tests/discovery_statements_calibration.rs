//! Coverage of the further statements of the structure-search routes: the order-MCMC,
//! structure-MCMC and CI-screened posteriors, the posterior over ancestral-graph classes, the
//! exact edge marginals and the class-collapsed bounded-lag template posterior.
//!
//! A structure posterior has no interval. What it states is mass, so each is measured by the
//! one event its mass makes a claim about:
//!
//! * A posterior over whole structures is scored by whether its smallest set of at least the
//!   stated mass holds the true structure, on a linear law whose structure is fixed and known.
//!   A Gaussian score cannot tell the members of a Markov equivalence class apart, so the DAG
//!   posteriors are scored through their class posterior against the class of the generating
//!   DAG (and order MCMC also as a DAG posterior under Laplace noise, where the DAG itself is
//!   identified); the posterior over ancestral-graph classes is scored against the partial
//!   ancestral graph of the generating DAG; the class-collapsed template posterior against the
//!   temporal class of the generating template.
//! * An edge marginal is a probability, not a set of structures. It is scored as a credible
//!   set for the edge indicator with the generating DAG drawn from the prior the search
//!   declares (uniform over the DAGs of the variables): a posterior that is right about the
//!   law of the data holds the indicator in its credible set at least as often as the level.
//!
//! Every set is held to at least nominal: a credible set of whole atoms holds at least its
//! stated mass. The set is encoded as the interval `[0, 1]` when it holds the truth and
//! `[2, 3]` when it does not, so a tally counts set coverage.
//!
//! The routes publish these statements in their payload only and key none at run time, so the
//! records name the construction here, as the family's first harness file does, with
//! validation `not_applicable`. A refused search (a sampler whose diagnostics gate withholds
//! the posterior included) is a skip; a search that runs and returns no posterior is a miss.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    reason = "test scaffolding indexes rows and atoms with small integers and keeps one design in one body"
)]

mod common;

use antecedent::discovery::{BayesianDiscoverParams, GraphMcmcSchedule, StaticDiscoverParams};
use antecedent::discovery_defaults::resolve_ci;
use antecedent::{CompiledExecution, DiscoveredStructure, DiscoverySpec, Study, StudyBuilder};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_discovery::{
    BackgroundKnowledge, CiSoftWeight, GraphPosterior, adjacency_mask_from_cpdag, set_edge,
};
use antecedent_graph::{Cpdag, DenseNodeId};
use common::calibration::{
    Construction, CoverageSide, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, gaussian,
    grid_n, map_replicates, n_sim, stream_seed, uniform,
};
use common::reported::GATE_LEVEL;

/// Validation mode of every record of this file: the routes have no support-matrix cell.
const NO_VALIDATION: &str = "not_applicable";

/// Rows of a static structure-search design.
const SEARCH_ROWS: usize = 150;

/// Rows of the Laplace-noise design, where the direction is read from the noise law.
const LAPLACE_ROWS: usize = 300;

/// Length of the series a temporal structure search reads.
const SEARCH_SERIES: usize = 200;

/// Schedule of every sampled DAG posterior: the schedule a search gets when it declares none.
const DAG_SCHEDULE: GraphMcmcSchedule =
    GraphMcmcSchedule { n_chains: 4, n_warmup: 500, n_draws: 1_000, thin: 1 };

/// Schedule of the sampler over ancestral graphs.
const PAG_SCHEDULE: GraphMcmcSchedule =
    GraphMcmcSchedule { n_chains: 4, n_warmup: 500, n_draws: 2_000, thin: 1 };

/// Schedule of the template search (unused when the templates are enumerated).
const TEMPLATE_SCHEDULE: GraphMcmcSchedule =
    GraphMcmcSchedule { n_chains: 4, n_warmup: 200, n_draws: 400, thin: 1 };

/// Interval method of a structure posterior scored as a credible set.
const STRUCTURE_SET: &str = "structure_credible_set";

/// Interval method of an edge marginal scored as a credible set for the edge indicator.
const EDGE_SET: &str = "edge_indicator_credible_set";

/// Interval method of a skeleton marginal scored as a credible set for the adjacency indicator.
const ADJACENCY_SET: &str = "adjacency_indicator_credible_set";

/// Why a structure credible set is held to at least nominal.
const STRUCTURE_SET_TARGET: &str = "true structure; a credible set of whole atoms holds at least \
                                    its stated mass";

/// Why an indicator credible set is held to at least nominal.
const INDICATOR_SET_TARGET: &str = "edge indicator of a generating DAG drawn from the declared \
                                    prior; the credible set is the more probable value alone or \
                                    both values, so it holds at least its stated mass";

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn table(columns: &[&str], values: &[Vec<f64>]) -> TabularData {
    TabularData::from_f64_columns(
        columns.iter().zip(values).map(|(name, col)| (*name, col.as_slice())),
    )
    .expect("the simulated columns are finite and aligned")
}

fn series(columns: &[&str], values: &[Vec<f64>]) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns(
        columns.iter().zip(values).map(|(name, col)| (*name, col.as_slice())),
        1,
    )
    .expect("the simulated columns are finite and aligned")
}

/// The construction of a statement a search publishes in its payload only.
fn payload_construction(
    graph_class: &str,
    modality: &str,
    estimator: &str,
    interval: &str,
    posterior: &str,
    functional: &str,
) -> Construction {
    Construction {
        query: "Discover".into(),
        graph_class: graph_class.into(),
        structure: "fixed".into(),
        modality: modality.into(),
        inference: "Bayesian".into(),
        estimator: estimator.into(),
        interval_method: interval.into(),
        se_kind: String::new(),
        dependence: "iid".into(),
        posterior: posterior.into(),
        functional: functional.into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
        unit: None,
    }
}

/// Scope facts of a payload statement measured on `rows` rows.
fn payload_scope(rows: usize, draws: Option<u32>) -> ScopeFacts {
    ScopeFacts {
        row_count: rows as u64,
        replicates_ok: None,
        posterior_draws: draws,
        unidentified_mass: 0.0,
    }
}

/// Retained draws of a sampled posterior under `schedule`.
fn draws(schedule: GraphMcmcSchedule) -> u32 {
    schedule.n_chains * schedule.n_draws / schedule.thin
}

/// Hold every tally to at least nominal coverage of `target`, printing every line before
/// failing on any of them.
fn gate_at_least(tallies: &[CoverageTally], target: &str) {
    let failures: Vec<String> = tallies
        .iter()
        .filter_map(|tally| {
            // The truth is a point, so either side names it.
            std::panic::catch_unwind(|| tally.assert_one_sided(CoverageSide::Lower, target))
                .err()
                .map(|panic| {
                    panic
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                        .unwrap_or_else(|| "coverage failure".into())
                })
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

/// What one replicate's search came to.
#[derive(Clone, Copy)]
enum Outcome<T> {
    /// The search was refused or could not be evaluated.
    Refused,
    /// The search ran and published no statement of the scored kind.
    Silent,
    /// Whether the scored set held the truth at the reported and at the gate level.
    Scored(T),
}

/// Scored tallies of one record at the reported and the gate level.
fn set_tallies(
    test: &'static str,
    dgp: &'static str,
    interval: &'static str,
) -> [CoverageTally; 2] {
    // One test can score more than one set kind (edge and adjacency), so each pair is
    // labelled by its kind: the tally sidecar keys a tally by name and level, and two kinds
    // under one name would overwrite each other's state.
    let key = RecordKey { test, dgp, interval };
    [
        CoverageTally::for_record(key, REPORTED_LEVEL).labelled(interval),
        CoverageTally::for_record(key, GATE_LEVEL).labelled(interval),
    ]
}

/// Record one replicate's set at both levels: `[0, 1]` when it holds the truth, `[2, 3]` when
/// it does not, nothing published as a miss and a refusal as a skip.
fn score_sets(
    tallies: &mut [CoverageTally; 2],
    outcome: Outcome<[bool; 2]>,
    construction: &Construction,
    scope: ScopeFacts,
) {
    for (index, tally) in tallies.iter_mut().enumerate() {
        match outcome {
            Outcome::Refused => tally.skip(),
            Outcome::Silent => {
                tally.bind_validation(construction, scope, NO_VALIDATION);
                tally.record(None, 1.0);
            }
            Outcome::Scored(holds) => {
                tally.bind_validation(construction, scope, NO_VALIDATION);
                tally.record(Some(if holds[index] { (0.0, 1.0) } else { (2.0, 3.0) }), 1.0);
            }
        }
    }
}

/// Run one search. The declaration is valid by design, so a study that does not build is a
/// test failure; a search the route refuses at run time is `None`.
fn search(test: &str, builder: StudyBuilder, seed: u64) -> Option<DiscoveredStructure> {
    let study = builder.build().unwrap_or_else(|error| panic!("{test}: {error}"));
    let execution = CompiledExecution::run(&study, &ExecutionContext::for_tests(seed)).ok()?;
    execution.result().discovery().map(|payload| payload.structure.clone())
}

/// One structure of a posterior: the adjacency word, the circle-mark word (zero without
/// marks) and the lag word (zero without a lag template).
type StructureKey = (u64, u64, u64);

/// The distinct structures of `posterior` with their pooled mass. A sampled posterior may
/// list one entry per draw; a credible set is over structures, so repeats are summed.
fn pooled_structures(posterior: &GraphPosterior) -> Option<(Vec<StructureKey>, Vec<f64>)> {
    let lags = posterior.lag_masks();
    let mut keys: Vec<StructureKey> = Vec::new();
    let mut mass: Vec<f64> = Vec::new();
    for (index, (atom, &weight)) in
        posterior.atoms().iter().zip(posterior.weights().iter()).enumerate()
    {
        let key = (
            atom.to_word()?,
            posterior.mark_atom(index).to_word()?,
            match lags {
                Some(lags) => lags.get(index)?.to_word()?,
                None => 0,
            },
        );
        if let Some(at) = keys.iter().position(|seen| *seen == key) {
            mass[at] += weight;
        } else {
            keys.push(key);
            mass.push(weight);
        }
    }
    Some((keys, mass))
}

/// Whether the smallest set of structures holding at least `level` of the posterior mass
/// contains `truth`: structures are taken in decreasing mass until the mass reaches `level`.
/// A truth the posterior does not list is outside every set.
fn credible_set_holds(
    keys: &[StructureKey],
    mass: &[f64],
    truth: StructureKey,
    level: f64,
) -> bool {
    let mut order: Vec<usize> = (0..mass.len()).collect();
    order.sort_by(|&a, &b| mass[b].total_cmp(&mass[a]));
    let mut taken = 0.0;
    for index in order {
        if keys[index] == truth {
            return true;
        }
        taken += mass[index];
        if taken >= level {
            return false;
        }
    }
    false
}

/// Score the structure posterior a search returned against `truth`.
fn structure_outcome(
    found: Option<DiscoveredStructure>,
    truth: StructureKey,
) -> Outcome<[bool; 2]> {
    match found {
        None => Outcome::Refused,
        Some(DiscoveredStructure::Posterior(posterior)) => match pooled_structures(&posterior) {
            Some((keys, mass)) => Outcome::Scored([
                credible_set_holds(&keys, &mass, truth, REPORTED_LEVEL),
                credible_set_holds(&keys, &mass, truth, GATE_LEVEL),
            ]),
            None => Outcome::Silent,
        },
        Some(_) => Outcome::Silent,
    }
}

/// One structure-posterior design: the record, the construction named for it and its scope.
struct SetDesign {
    test: &'static str,
    dgp: &'static str,
    construction: Construction,
    scope: ScopeFacts,
    seed_base: u64,
    truth: StructureKey,
}

/// Measure the credible sets of the posterior `run(seed)` returns over the replicates.
fn measure_sets(design: &SetDesign, run: impl Fn(u64) -> Option<DiscoveredStructure> + Sync) {
    let mut tallies = set_tallies(design.test, design.dgp, STRUCTURE_SET);
    let outcomes = map_replicates(n_sim(), |rep| {
        structure_outcome(run(stream_seed(design.seed_base, rep)), design.truth)
    });
    for outcome in outcomes {
        score_sets(&mut tallies, outcome, &design.construction, design.scope);
    }
    gate_at_least(&tallies, STRUCTURE_SET_TARGET);
}

// ------------------------------------------------------------------ data

/// `z -> t -> y` with Gaussian noise and no direct edge: `t = 0.8 z + e`, `y = 1.5 t + e'`.
fn pure_chain_rows(rows: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut z, mut t, mut y) =
        (Vec::with_capacity(rows), Vec::with_capacity(rows), Vec::with_capacity(rows));
    for _ in 0..rows {
        let zi = g();
        let ti = 0.8 * zi + g();
        let yi = 1.5 * ti + g();
        z.push(zi);
        t.push(ti);
        y.push(yi);
    }
    table(&["z", "t", "y"], &[z, t, y])
}

/// `x -> y <- z` with Gaussian noise and independent causes: `y = 0.9 x + 0.9 z + e`. The
/// columns are `x`, `y`, `z`.
fn collider_rows(rows: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut x, mut y, mut z) =
        (Vec::with_capacity(rows), Vec::with_capacity(rows), Vec::with_capacity(rows));
    for _ in 0..rows {
        let xi = g();
        let zi = g();
        x.push(xi);
        z.push(zi);
        y.push(0.9 * xi + 0.9 * zi + g());
    }
    table(&["x", "y", "z"], &[x, y, z])
}

/// `x -> y` with standard Laplace noise: `y = 1.5 x + e`.
fn laplace_pair_rows(rows: usize, seed: u64) -> TabularData {
    let mut unit = uniform(seed, 0x1A91);
    let mut laplace = || {
        let u = unit() - 0.5;
        -u.signum() * (1.0 - 2.0 * u.abs()).max(1e-12).ln()
    };
    let (mut x, mut y) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
    for _ in 0..rows {
        let xi = laplace();
        x.push(xi);
        y.push(1.5 * xi + laplace());
    }
    table(&["x", "y"], &[x, y])
}

/// `x_t ~ N(0, 1)`, `y_t = 2 x_{t-1} + 0.5 e`, started from a draw of the stationary law.
fn lag_chain_series(steps: usize, seed: u64) -> TimeSeriesData {
    let mut g = gaussian(seed);
    let (mut x, mut y) = (Vec::with_capacity(steps), Vec::with_capacity(steps));
    let mut previous = g();
    for _ in 0..steps {
        let xi = g();
        x.push(xi);
        y.push(2.0 * previous + 0.5 * g());
        previous = xi;
    }
    series(&["x", "y"], &[x, y])
}

/// `x_t ~ N(0, 1)`, `y_t = 0.8 x_t + 0.6 e`, independent across steps: one contemporaneous
/// edge and no lagged one.
fn contemporaneous_pair_series(steps: usize, seed: u64) -> TimeSeriesData {
    let mut g = gaussian(seed);
    let (mut x, mut y) = (Vec::with_capacity(steps), Vec::with_capacity(steps));
    for _ in 0..steps {
        let xi = g();
        x.push(xi);
        y.push(0.8 * xi + 0.6 * g());
    }
    series(&["x", "y"], &[x, y])
}

/// The link of each unordered pair `(0, 1)`, `(0, 2)`, `(1, 2)` of a three-variable DAG:
/// 0 absent, 1 from the lower index to the higher, 2 from the higher to the lower.
type Links = [u8; 3];

/// The 25 DAGs on three labelled variables: the 27 link assignments less the two cycles.
fn three_variable_dags() -> Vec<Links> {
    let mut dags = Vec::with_capacity(25);
    for code in 0..27_u8 {
        let links: Links = [code % 3, (code / 3) % 3, code / 9];
        // 0 -> 1 -> 2 -> 0 and 0 -> 2 -> 1 -> 0.
        if links != [1, 2, 1] && links != [2, 1, 2] {
            dags.push(links);
        }
    }
    dags
}

/// Rows of a linear-Gaussian model on a DAG drawn uniformly from [`three_variable_dags`],
/// with the DAG's links. Each edge coefficient has a magnitude uniform on `[0.5, 1.5]` and a
/// fair sign, and each noise term is standard normal.
fn prior_dag_rows(rows: usize, seed: u64) -> (TabularData, Links) {
    const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];
    let mut unit = uniform(seed, 0x1A92);
    let dags = three_variable_dags();
    let links = dags[((unit() * dags.len() as f64) as usize).min(dags.len() - 1)];
    // `weight[to][from]`, zero where there is no edge.
    let mut weight = [[0.0_f64; 3]; 3];
    for (&(low, high), &link) in PAIRS.iter().zip(&links) {
        let magnitude = 0.5 + unit();
        let coefficient = if unit() < 0.5 { -magnitude } else { magnitude };
        match link {
            1 => weight[high][low] = coefficient,
            2 => weight[low][high] = coefficient,
            _ => {}
        }
    }
    // A causal order: a node is placed once every parent is.
    let mut order: Vec<usize> = Vec::with_capacity(3);
    while order.len() < 3 {
        let next = (0..3)
            .find(|node| {
                !order.contains(node)
                    && (0..3).all(|from| weight[*node][from] == 0.0 || order.contains(&from))
            })
            .expect("the drawn graph is acyclic");
        order.push(next);
    }
    let mut g = gaussian(stream_seed(seed, 0x1A93));
    let mut columns = (0..3).map(|_| Vec::with_capacity(rows)).collect::<Vec<_>>();
    for _ in 0..rows {
        let mut value = [0.0_f64; 3];
        for &node in &order {
            value[node] = (0..3).map(|from| weight[node][from] * value[from]).sum::<f64>() + g();
        }
        for (column, x) in columns.iter_mut().zip(value) {
            column.push(x);
        }
    }
    (table(&["a", "b", "c"], &columns), links)
}

// ------------------------------------------------------------------ truths

/// The Markov equivalence class of `0 - 1 - 2` with no collider: both edges undirected.
fn chain_class() -> u64 {
    let mut class = Cpdag::with_variables(3);
    class.insert_undirected(d(0), d(1)).expect("a fresh edge");
    class.insert_undirected(d(1), d(2)).expect("a fresh edge");
    adjacency_mask_from_cpdag(&class).expect("three variables pack into one word")
}

/// The class of one undirected edge between two variables.
fn pair_class() -> u64 {
    let mut class = Cpdag::with_variables(2);
    class.insert_undirected(d(0), d(1)).expect("a fresh edge");
    adjacency_mask_from_cpdag(&class).expect("two variables pack into one word")
}

/// The partial ancestral graph of the collider `0 -> 1 <- 2`: an arrowhead at the collider on
/// both edges and a circle at each cause, since a latent common cause of a cause and the
/// collider leaves every independence unchanged. A circle-arrow edge `i o-> j` is the
/// adjacency bit `(i, j)` with the mark bit `(i, j)`.
fn collider_pag() -> StructureKey {
    let edges = set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 1, true);
    (edges, edges, 0)
}

/// The partial ancestral graph of the chain `0 - 1 - 2` with no collider: every endpoint is a
/// circle, which is both adjacency bits and both mark bits of each edge.
fn chain_pag() -> StructureKey {
    let mut edges = 0;
    for (a, b) in [(0, 1), (1, 2)] {
        edges = set_edge(set_edge(edges, 3, a, b, true), 3, b, a, true);
    }
    (edges, edges, 0)
}

fn three() -> Vec<VariableId> {
    vec![v(0), v(1), v(2)]
}

/// Scope of a three-variable tabular design at this run's grid point.
fn search_rows() -> usize {
    grid_n(SEARCH_ROWS)
}

// ------------------------------------------------------------------ DAG posteriors

/// The class posterior of the order-MCMC posterior of three variables, against the class of
/// the generating chain. The sampler runs under its diagnostics gate, as a search does by
/// default: a posterior the gate withholds is a refused replicate.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn discovery_static_order_mcmc_class_posterior_credible_set_coverage() {
    const TEST: &str = "discovery_static_order_mcmc_class_posterior_credible_set_coverage";
    let design = SetDesign {
        test: TEST,
        dgp: "pure_chain_rows",
        construction: payload_construction(
            "Cpdag",
            "tabular",
            "discovery.order_mcmc",
            STRUCTURE_SET,
            "order_mcmc.equivalence_classes.gaussian_bic.mcmc_4x1000",
            "discover.class_posterior",
        ),
        scope: payload_scope(search_rows(), Some(draws(DAG_SCHEDULE))),
        seed_base: 0xD15_0001,
        truth: (chain_class(), 0, 0),
    };
    measure_sets(&design, |seed| {
        let spec = DiscoverySpec::order_mcmc(BayesianDiscoverParams::default(), DAG_SCHEDULE, true)
            .equivalence_classes();
        search(
            TEST,
            Study::tabular(pure_chain_rows(search_rows(), seed)).discovery(spec, three()),
            seed,
        )
    });
}

/// The order-MCMC posterior over DAGs of two variables under the Laplace-noise score, against
/// the generating DAG `x -> y`: with Laplace noise the direction is identified, so the DAG
/// itself is the truth.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn discovery_static_order_mcmc_laplace_dag_posterior_credible_set_coverage() {
    const TEST: &str = "discovery_static_order_mcmc_laplace_dag_posterior_credible_set_coverage";
    let rows = grid_n(LAPLACE_ROWS);
    let design = SetDesign {
        test: TEST,
        dgp: "laplace_pair_rows",
        construction: payload_construction(
            "Dag",
            "tabular",
            "discovery.order_mcmc",
            STRUCTURE_SET,
            "order_mcmc.laplace_bic.mcmc_4x1000",
            "discover.dag_posterior",
        ),
        scope: payload_scope(rows, Some(draws(DAG_SCHEDULE))),
        seed_base: 0xD15_0002,
        truth: (set_edge(0, 2, 0, 1, true), 0, 0),
    };
    measure_sets(&design, |seed| {
        let params = BayesianDiscoverParams {
            score_family: antecedent_state::GraphScoreFamily::LaplaceBic,
            ..BayesianDiscoverParams::default()
        };
        search(
            TEST,
            Study::tabular(laplace_pair_rows(rows, seed))
                .discovery(DiscoverySpec::order_mcmc(params, DAG_SCHEDULE, true), vec![v(0), v(1)]),
            seed,
        )
    });
}

/// The class posterior of the structure-MCMC posterior of three variables, against the class
/// of the generating chain.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn discovery_static_structure_mcmc_class_posterior_credible_set_coverage() {
    const TEST: &str = "discovery_static_structure_mcmc_class_posterior_credible_set_coverage";
    let design = SetDesign {
        test: TEST,
        dgp: "pure_chain_rows",
        construction: payload_construction(
            "Cpdag",
            "tabular",
            "discovery.structure_mcmc",
            STRUCTURE_SET,
            "structure_mcmc.equivalence_classes.gaussian_bic.mcmc_4x1000",
            "discover.class_posterior",
        ),
        scope: payload_scope(search_rows(), Some(draws(DAG_SCHEDULE))),
        seed_base: 0xD15_0003,
        truth: (chain_class(), 0, 0),
    };
    measure_sets(&design, |seed| {
        let spec = DiscoverySpec::structure_mcmc(BayesianDiscoverParams::default(), DAG_SCHEDULE)
            .equivalence_classes();
        search(
            TEST,
            Study::tabular(pure_chain_rows(search_rows(), seed)).discovery(spec, three()),
            seed,
        )
    });
}

/// The class posterior of the CI-screened posterior of three variables, against the class of
/// the generating chain. The screen is the partial-correlation skeleton at level 0.05 with no
/// soft weight; a pair the screen drops is never proposed, so a screen that drops a true edge
/// leaves the true class outside every set.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn discovery_static_ci_screened_class_posterior_credible_set_coverage() {
    const TEST: &str = "discovery_static_ci_screened_class_posterior_credible_set_coverage";
    let design = SetDesign {
        test: TEST,
        dgp: "pure_chain_rows",
        construction: payload_construction(
            "Cpdag",
            "tabular",
            "discovery.ci_screened_posterior",
            STRUCTURE_SET,
            "ci_screened_posterior.parcorr_0_05.equivalence_classes.gaussian_bic.mcmc_4x1000",
            "discover.class_posterior",
        ),
        scope: payload_scope(search_rows(), Some(draws(DAG_SCHEDULE))),
        seed_base: 0xD15_0004,
        truth: (chain_class(), 0, 0),
    };
    measure_sets(&design, |seed| {
        let screen = StaticDiscoverParams {
            alpha: 0.05,
            max_cond_size: 1,
            fdr: None,
            ci: resolve_ci("parcorr", None).expect("the partial-correlation test resolves"),
            screen_pc: false,
            max_subset: None,
            knowledge: BackgroundKnowledge::default(),
        };
        let spec = DiscoverySpec::ci_screened_posterior(
            BayesianDiscoverParams::default(),
            screen,
            DAG_SCHEDULE,
            CiSoftWeight::None,
            "parcorr",
        )
        .equivalence_classes();
        search(
            TEST,
            Study::tabular(pure_chain_rows(search_rows(), seed)).discovery(spec, three()),
            seed,
        )
    });
}

// ------------------------------------------------------------------ ancestral-graph classes

/// The construction of a posterior over ancestral-graph classes.
fn pag_construction(estimator: &str, posterior: &str) -> Construction {
    payload_construction(
        "Pag",
        "tabular",
        estimator,
        STRUCTURE_SET,
        posterior,
        "discover.pag_posterior",
    )
}

/// The exact posterior over ancestral-graph classes of three variables, against the partial
/// ancestral graph of the generating collider `x o-> y <-o z`.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn discovery_static_exact_pag_posterior_collider_credible_set_coverage() {
    const TEST: &str = "discovery_static_exact_pag_posterior_collider_credible_set_coverage";
    let design = SetDesign {
        test: TEST,
        dgp: "collider_rows",
        construction: pag_construction(
            "discovery.exact_pag_posterior",
            "exact_pag_posterior.gaussian_mag_bic",
        ),
        scope: payload_scope(search_rows(), None),
        seed_base: 0xD15_0005,
        truth: collider_pag(),
    };
    measure_sets(&design, |seed| {
        search(
            TEST,
            Study::tabular(collider_rows(search_rows(), seed)).discovery(
                DiscoverySpec::exact_pag_posterior(BayesianDiscoverParams::default()),
                three(),
            ),
            seed,
        )
    });
}

/// The exact posterior over ancestral-graph classes of three variables, against the partial
/// ancestral graph of the generating chain `z o-o t o-o y`.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn discovery_static_exact_pag_posterior_chain_credible_set_coverage() {
    const TEST: &str = "discovery_static_exact_pag_posterior_chain_credible_set_coverage";
    let design = SetDesign {
        test: TEST,
        dgp: "pure_chain_rows",
        construction: pag_construction(
            "discovery.exact_pag_posterior",
            "exact_pag_posterior.gaussian_mag_bic",
        ),
        scope: payload_scope(search_rows(), None),
        seed_base: 0xD15_0006,
        truth: chain_pag(),
    };
    measure_sets(&design, |seed| {
        search(
            TEST,
            Study::tabular(pure_chain_rows(search_rows(), seed)).discovery(
                DiscoverySpec::exact_pag_posterior(BayesianDiscoverParams::default()),
                three(),
            ),
            seed,
        )
    });
}

/// The sampled posterior over ancestral-graph classes of three variables, against the partial
/// ancestral graph of the generating collider.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn discovery_static_pag_mcmc_posterior_collider_credible_set_coverage() {
    const TEST: &str = "discovery_static_pag_mcmc_posterior_collider_credible_set_coverage";
    let design = SetDesign {
        test: TEST,
        dgp: "collider_rows",
        construction: pag_construction(
            "discovery.pag_mcmc",
            "pag_mcmc.gaussian_mag_bic.mcmc_4x2000",
        ),
        scope: payload_scope(search_rows(), Some(draws(PAG_SCHEDULE))),
        seed_base: 0xD15_0007,
        truth: collider_pag(),
    };
    measure_sets(&design, |seed| {
        search(
            TEST,
            Study::tabular(collider_rows(search_rows(), seed)).discovery(
                DiscoverySpec::pag_mcmc(BayesianDiscoverParams::default(), PAG_SCHEDULE),
                three(),
            ),
            seed,
        )
    });
}

// ------------------------------------------------------------------ edge marginals

/// Whether the smallest credible set of mass `level` over `{absent, present}` holds the truth:
/// the set is the more probable value alone once its mass reaches `level`, else both values.
fn indicator_set_holds(probability: f64, present: bool, level: f64) -> bool {
    let truth = if present { probability } else { 1.0 - probability };
    1.0 - truth < level
}

/// The exact edge marginals of three variables with the generating DAG drawn from the declared
/// uniform prior over DAGs: the marginal `P(a -> b)` as a credible set for that edge's
/// indicator, and the skeleton marginal `P(a -> b) + P(b -> a)` as a credible set for the
/// adjacency of the pair. The generating law treats the three labels alike, so the one scored
/// pair stands for every pair without scoring dependent pairs of one dataset together.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn discovery_static_exact_edge_marginals_indicator_credible_set_coverage() {
    const TEST: &str = "discovery_static_exact_edge_marginals_indicator_credible_set_coverage";
    const DGP: &str = "prior_dag_rows";
    let rows = search_rows();
    let scope = payload_scope(rows, None);
    let construction = |interval: &str, functional: &str| {
        payload_construction(
            "Dag",
            "tabular",
            "discovery.exact_dag_marginals",
            interval,
            "exact_dag_marginals.gaussian_bic.uniform_dag_prior",
            functional,
        )
    };
    let edge = construction(EDGE_SET, "discover.edge_marginal");
    let adjacency = construction(ADJACENCY_SET, "discover.skeleton_marginal");
    let mut edge_tallies = set_tallies(TEST, DGP, EDGE_SET);
    let mut adjacency_tallies = set_tallies(TEST, DGP, ADJACENCY_SET);
    let outcomes = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0xD15_0008, rep);
        let (data, links) = prior_dag_rows(rows, seed);
        let spec = DiscoverySpec::exact_dag_marginals(BayesianDiscoverParams::default());
        match search(TEST, Study::tabular(data).discovery(spec, three()), seed) {
            None => (Outcome::Refused, Outcome::Refused),
            Some(DiscoveredStructure::EdgeMarginals(marginals)) => {
                // Row-major `from * n + to` over three variables.
                let forward = marginals.edge_marginals[1];
                let backward = marginals.edge_marginals[3];
                let sets = |probability: f64, present: bool| {
                    Outcome::Scored([
                        indicator_set_holds(probability, present, REPORTED_LEVEL),
                        indicator_set_holds(probability, present, GATE_LEVEL),
                    ])
                };
                (sets(forward, links[0] == 1), sets(forward + backward, links[0] != 0))
            }
            Some(_) => (Outcome::Silent, Outcome::Silent),
        }
    });
    for (edge_outcome, adjacency_outcome) in outcomes {
        score_sets(&mut edge_tallies, edge_outcome, &edge, scope);
        score_sets(&mut adjacency_tallies, adjacency_outcome, &adjacency, scope);
    }
    let tallies: Vec<CoverageTally> = edge_tallies.into_iter().chain(adjacency_tallies).collect();
    gate_at_least(&tallies, INDICATOR_SET_TARGET);
}

// ------------------------------------------------------------------ temporal classes

/// The construction of the class-collapsed template posterior of two variables with one lag.
fn template_class_construction() -> Construction {
    payload_construction(
        "TemporalCpdag",
        "series",
        "discovery.dbn_posterior",
        STRUCTURE_SET,
        "dbn_posterior.equivalence_classes.gaussian_bic.mcmc_4x400",
        "discover.dbn_template_class_posterior",
    )
}

fn template_class_search(
    test: &str,
    data: TimeSeriesData,
    seed: u64,
) -> Option<DiscoveredStructure> {
    let spec = DiscoverySpec::dbn_posterior(
        BayesianDiscoverParams::default(),
        1,
        false,
        TEMPLATE_SCHEDULE,
    )
    .equivalence_classes();
    search(test, Study::series(data).discovery(spec, vec![v(0), v(1)]), seed)
}

/// The class-collapsed template posterior of two variables with one lag, against the temporal
/// class of the generating template: no contemporaneous edge and the lagged edge
/// `x_{t-1} -> y_t`, which time orients.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn discovery_temporal_dbn_class_posterior_lagged_edge_credible_set_coverage() {
    const TEST: &str = "discovery_temporal_dbn_class_posterior_lagged_edge_credible_set_coverage";
    let steps = grid_n(SEARCH_SERIES);
    let design = SetDesign {
        test: TEST,
        dgp: "lag_chain_series",
        construction: template_class_construction(),
        scope: payload_scope(steps, Some(draws(TEMPLATE_SCHEDULE))),
        seed_base: 0xD15_0009,
        // Bit `(lag - 1) * n^2 + from * n + to` of the lag mask with two variables.
        truth: (0, 0, 1 << 1),
    };
    measure_sets(&design, |seed| template_class_search(TEST, lag_chain_series(steps, seed), seed));
}

/// The class-collapsed template posterior of two variables with one lag, against the temporal
/// class of a template with one contemporaneous edge and no lagged edge. Nothing orients that
/// edge, so the class leaves it undirected and pools the two templates that differ only in
/// its direction: the design where the collapse changes the atoms.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn discovery_temporal_dbn_class_posterior_contemporaneous_edge_credible_set_coverage() {
    const TEST: &str =
        "discovery_temporal_dbn_class_posterior_contemporaneous_edge_credible_set_coverage";
    let steps = grid_n(SEARCH_SERIES);
    let design = SetDesign {
        test: TEST,
        dgp: "contemporaneous_pair_series",
        construction: template_class_construction(),
        scope: payload_scope(steps, Some(draws(TEMPLATE_SCHEDULE))),
        seed_base: 0xD15_000A,
        truth: (pair_class(), 0, 0),
    };
    measure_sets(&design, |seed| {
        template_class_search(TEST, contemporaneous_pair_series(steps, seed), seed)
    });
}

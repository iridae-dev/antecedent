//! Repeated-sampling coverage of the general identification functional estimator
//! (`functional.effect`) over a graph posterior of partial-ancestral-graph atoms: the natural
//! mediation decomposition, the natural direct effect as a nested counterfactual, and the
//! natural indirect effect on the unrolled window of a temporal template.
//!
//! One test measures one coordinate: a query, a PAG class, an inference mode and the
//! graph-posterior structure family. The posterior holds one atom, and the validation modes
//! (`none`, `cheap`, `full`) are sibling executions of that coordinate: the driver measures the
//! first on every replicate and requires every sibling to report its interval bit for bit.
//!
//! The laws are the exact finite populations and stationary processes of the known-truth
//! fixtures: `conformance/estimate/mediation_mixed_graphs` (`markov_parents`),
//! `conformance/estimate/cross_world_class_truth` (`mediated_two_parents`) and
//! `conformance/estimate/temporal_window_truth` (`mediation`). A replicate draws its rows
//! uniformly from the enumerated table, or simulates the structural process itself, so the
//! population value of each identified functional is the fixture's exact truth. Every candidate
//! functional a PAG atom stands for evaluates to that truth on the law (the generating DAG is
//! Markov to every candidate, and the confounding a candidate adds touches only a root's edge),
//! so the enumeration-weighted mean the atom reports is the truth's estimator.
//!
//! The atom is a PAG, and a PAG's candidate functionals differ as expressions (the ADMGs a MAG
//! completion stands for add a confounded root edge), so its execution publishes the class
//! identified set and no standard error or posterior: the point is the enumeration-weighted mean
//! of the candidates. One contributing atom carries the whole conditional mixture, so the
//! graph posterior publishes its atom's identified-set interval
//! (`structural_response.identified_set_interval`, bound as `identified_set`) exactly as the
//! given-structure coordinates of `grid_functional_effect_calibration` do. A Frequentist
//! execution publishes the Imbens-Manski interval from every candidate refitted on shared
//! bootstrap resamples at 0.95 (circular blocks of consecutive windows on a temporal template);
//! a Bayesian one the product-posterior envelope of the candidates' Bayesian-bootstrap
//! posteriors at 0.90. Each is published at that one level, so the driver scores the one
//! published interval at its own level. Every candidate evaluates to the truth on the law, so the
//! population identified set is the point at the truth and the record is gated one-sided:
//! coverage at least the level, no upper band.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    reason = "test scaffolding compares exact constants, indexes small tables and keeps one design table in one body"
)]

mod common;
#[path = "common/grid_cell.rs"]
mod grid_cell;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::adjacency_mask::{AdjacencyMask, adjacency_masks_from_pag, set_edge};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_graph::{DenseNodeId, Pag};
use antecedent_prob::InferenceDiagnostics;
use common::calibration::{
    Construction, CoverageSide, CoverageTally, RecordKey, ScopeFacts, map_replicates, n_sim,
    stream_seed, uniform,
};
use common::calibration_bind::constructions;
use grid_cell::SIBLING_REPLICATES;
use serde_json::Value;

/// Posterior draws of a Bayesian execution.
const DRAWS: usize = 400;

/// Estimator every coordinate of this file executes.
const FUNCTIONAL: &str = "functional.effect";

/// The validation modes of a graph posterior: the base and its siblings.
const SUITES: [RefuteSuite; 3] = [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full];

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

// ----------------------------------------------------------------- populations

/// An exact finite population: the rows of an enumerated law, each equally likely.
#[derive(Debug, Clone)]
struct Population {
    names: Vec<String>,
    /// Column-major values, one row per population member.
    columns: Vec<Vec<f64>>,
}

impl Population {
    /// The population of `names` holding `count` copies of each `(row, count)` cell.
    fn from_cells(names: &[&str], cells: impl IntoIterator<Item = (Vec<f64>, u64)>) -> Self {
        let mut columns = vec![Vec::new(); names.len()];
        for (row, count) in cells {
            assert_eq!(row.len(), names.len(), "one value per column");
            for (column, value) in columns.iter_mut().zip(&row) {
                column.extend(std::iter::repeat_n(*value, count as usize));
            }
        }
        assert!(!columns[0].is_empty(), "an empty population");
        Self { names: names.iter().map(|name| (*name).to_owned()).collect(), columns }
    }

    /// The population of a fixture case with `columns` (names) and `contingency_table`
    /// (objects holding a value per column and a `count`).
    fn from_contingency(case: &Value) -> Self {
        let names: Vec<&str> =
            case["columns"].as_array().unwrap().iter().map(|name| name.as_str().unwrap()).collect();
        let cells = case["contingency_table"].as_array().unwrap().iter().map(|cell| {
            (
                names.iter().map(|name| cell[*name].as_f64().unwrap()).collect(),
                cell["count"].as_u64().unwrap(),
            )
        });
        Self::from_cells(&names, cells)
    }

    fn size(&self) -> usize {
        self.columns[0].len()
    }

    /// Column index (the study's variable id) of `name`.
    fn index(&self, name: &str) -> u32 {
        let index = self
            .names
            .iter()
            .position(|column| column == name)
            .unwrap_or_else(|| panic!("no column {name}"));
        u32::try_from(index).unwrap()
    }

    /// `n` rows drawn independently and uniformly from the population.
    fn sample(&self, n: usize, seed: u64) -> TabularData {
        let mut u01 = uniform(seed, 0x6E1D_F00D);
        let size = self.size();
        let rows: Vec<usize> =
            (0..n).map(|_| ((u01() * size as f64) as usize).min(size - 1)).collect();
        let sampled: Vec<Vec<f64>> = self
            .columns
            .iter()
            .map(|column| rows.iter().map(|&row| column[row]).collect())
            .collect();
        TabularData::from_f64_columns(
            self.names
                .iter()
                .zip(&sampled)
                .map(|(name, values)| (name.as_str(), values.as_slice())),
        )
        .unwrap()
    }
}

// ------------------------------------------------------------------ structures

fn directed_mask(nodes: usize, edges: &[(u32, u32)]) -> AdjacencyMask {
    edges.iter().fold(AdjacencyMask::EMPTY, |mask, &(from, to)| {
        set_edge(mask, nodes, from as usize, to as usize, true)
    })
}

fn pag_of(nodes: u32, directed: &[(u32, u32)]) -> Pag {
    let mut graph = Pag::with_variables(nodes);
    for &(from, to) in directed {
        graph.insert_directed(d(from), d(to)).unwrap();
    }
    graph
}

fn sampling() -> SamplingSummary {
    SamplingSummary {
        ess: 1.0,
        diagnostics: InferenceDiagnostics::analytic("single_atom_functional"),
        rejected_invalid: 0,
    }
}

/// A graph posterior holding all its mass on the static PAG `graph`.
fn pag_posterior(graph: &Pag) -> GraphPosterior {
    let (adjacency, marks) = adjacency_masks_from_pag(graph).unwrap();
    GraphPosterior::from_atoms(
        graph.node_count(),
        vec![1.0],
        vec![adjacency],
        vec![0.0; graph.node_count() * graph.node_count()],
        vec![0.0; graph.node_count() * graph.node_count()],
        sampling(),
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Pag)
    .with_mark_atoms(vec![marks])
    .unwrap()
}

/// A graph posterior holding all its mass on one stationary lagged PAG template: the
/// contemporaneous edges and the lag-1 directed edges `(lag, from, to)` of a lagged `Pag` atom.
fn lagged_pag_posterior(
    nodes: usize,
    contemporaneous: &[(u32, u32)],
    lagged: &[(u32, usize, usize)],
) -> GraphPosterior {
    let lag_mask =
        lagged.iter().fold(antecedent_discovery::LagMask::EMPTY, |word, &(lag, from, to)| {
            antecedent_discovery::with_lagged_edge(word, nodes, lag, from, to, true).unwrap()
        });
    GraphPosterior::from_atoms(
        nodes,
        vec![1.0],
        vec![directed_mask(nodes, contemporaneous)],
        vec![0.0; nodes * nodes],
        vec![0.0; nodes * nodes],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("single_atom_temporal_window"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Pag)
    .with_lagged_marginals(1, vec![0.0; nodes * nodes])
    .unwrap()
    .with_lag_masks(vec![lag_mask])
    .unwrap()
}

// ----------------------------------------------------------------------- modes

/// The inference mode of a coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Frequentist,
    Bayesian,
}

impl Mode {
    fn inference(self) -> InferenceMode {
        match self {
            Self::Frequentist => InferenceMode::Frequentist,
            Self::Bayesian => InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(DRAWS)),
        }
    }

    /// Level a class identified set is published at: the Imbens-Manski interval of a
    /// Frequentist execution, the product-posterior envelope of a Bayesian one.
    fn identified_set_level(self) -> f64 {
        match self {
            Self::Frequentist => 0.95,
            Self::Bayesian => 0.9,
        }
    }
}

/// `IntervalMethod::as_str()` of a class identified-set interval.
const IDENTIFIED_SET: &str = "identified_set";

/// The identified-set interval of `result`, checked to be published at `level`; `None` when the
/// candidates agreed on this sample and the result published a point.
fn identified_set(result: &StudyResult, level: f64) -> Option<(f64, f64)> {
    let set = result.structural_response.as_ref()?.identified_set_interval.as_ref()?;
    assert!((set.level - level).abs() < 1e-12, "identified set published at {}", set.level);
    Some((set.lower, set.upper))
}

// ------------------------------------------------------------------- execution

/// The coordinate one test measures.
struct Coordinate {
    /// The test's own name.
    test: &'static str,
    /// The generator's name.
    dgp: &'static str,
    /// Seed family of the test's replicates.
    family: u64,
}

/// One execution's identified set and the runtime's binding of it.
type SetRun = (Option<(f64, f64)>, Option<(Construction, ScopeFacts, &'static str)>);

/// Measure the identified set of `coordinate` against `truth`: `study(seed, suite)` builds one
/// execution on the replicate's dataset (which must depend on `seed` only) under validation mode
/// `suite`; the first of [`SUITES`] is the base and the others are its siblings.
///
/// The base is scored on every replicate; every sibling is executed on the first
/// [`SIBLING_REPLICATES`] replicates, must publish the base interval bit for bit, and gets its own
/// record bound from its own executions' match key and validation mode. A replicate whose
/// candidates agreed publishes no set and is scored as a miss.
fn measure_coordinate(
    coordinate: &Coordinate,
    mode: Mode,
    truth: f64,
    study: impl Fn(u64, RefuteSuite) -> Study + Sync,
) {
    let level = mode.identified_set_level();
    let test = coordinate.test;
    let execute = |rep: u64, variant: usize| -> SetRun {
        let seed = stream_seed(coordinate.family, rep);
        let suite = SUITES[variant];
        let study = study(seed, suite);
        let result = study
            .run(&ExecutionContext::for_tests(seed))
            .unwrap_or_else(|e| panic!("{test} {suite:?}: {e}"));
        assert_eq!(
            result.logical_plan.estimator.as_deref(),
            Some(FUNCTIONAL),
            "{test} {suite:?}: estimator target"
        );
        let interval = identified_set(&result, level);
        let bound = constructions(&study.inspect().expect("inspect"), &result)
            .into_iter()
            .find(|(construction, _, _)| construction.interval_method == IDENTIFIED_SET);
        assert_eq!(
            interval.is_some(),
            bound.is_some(),
            "{test} {suite:?}: the runtime binds the identified set exactly when it is published"
        );
        (interval, bound)
    };
    let bits = |interval: Option<(f64, f64)>| interval.map(|(lo, hi)| (lo.to_bits(), hi.to_bits()));
    let runs = map_replicates(n_sim(), |rep| {
        let base = execute(rep, 0);
        let siblings: Vec<SetRun> = if rep < u64::from(SIBLING_REPLICATES) {
            (1..SUITES.len())
                .map(|variant| {
                    let sibling = execute(rep, variant);
                    assert_eq!(
                        bits(sibling.0),
                        bits(base.0),
                        "{test}: variant {variant} does not report the base identified set on \
                         replicate {rep}"
                    );
                    sibling
                })
                .collect()
        } else {
            Vec::new()
        };
        (base, siblings)
    });

    // Each sibling's match key, from its own executions; stable across them.
    let keys: Vec<(Construction, &'static str)> = (1..SUITES.len())
        .map(|variant| {
            let mut seen: Option<(Construction, &'static str)> = None;
            for (_, siblings) in &runs {
                if let Some((_, Some((construction, _, validation)))) = siblings.get(variant - 1) {
                    let found = (construction.clone(), *validation);
                    match &seen {
                        Some(prior) => assert_eq!(prior, &found, "{test}: sibling match key"),
                        None => seen = Some(found),
                    }
                }
            }
            seen.unwrap_or_else(|| panic!("{test}: variant {variant} published no identified set"))
        })
        .collect();

    let tallies: Vec<CoverageTally> = (0..SUITES.len())
        .map(|variant| {
            let key = RecordKey { test, dgp: coordinate.dgp, interval: IDENTIFIED_SET };
            let mut tally = CoverageTally::for_record(key, level);
            if variant > 0 {
                tally = tally.named(format!("{test}#v{variant}"));
            }
            for ((interval, bound), _) in &runs {
                if let Some((construction, scope, validation)) = bound {
                    match variant {
                        0 => tally.bind_validation(construction, *scope, validation),
                        v => tally.bind_validation(&keys[v - 1].0, *scope, keys[v - 1].1),
                    }
                }
                tally.record(*interval, truth);
            }
            tally
        })
        .collect();
    let failures: Vec<String> = tallies
        .iter()
        .filter_map(|tally| {
            std::panic::catch_unwind(|| {
                tally.assert_one_sided(
                    CoverageSide::Lower,
                    "the population identified set, the point at the effect every candidate \
                     functional takes on the law",
                );
            })
            .err()
            .map(|e| {
                e.downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                    .unwrap_or_else(|| "coverage failure".into())
            })
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

// ----------------------------------------------------- natural mediation on a PAG
//
// MediationEffect: the natural direct effect of `t` on `y` through `m`, on the `markov_parents`
// law (`c -> t`, `w -> m`, `t -> m`, `t -> y`, `m -> y`, a DAG over the observed columns
// `c, w, t, m, y`) and its PAG with every edge directed. The edges out of the roots `c` and
// `w` are invisible, so the PAG's MAG completion stands for ADMGs that add `c <-> t` or
// `w <-> m`; their functionals differ as expressions and take the enumerated natural direct
// effect `5/32` on the table. The graph posterior holds the one PAG atom.

/// Base sample size of the `markov_parents` design: 256 rows, the rarest cell holds one
/// (`p = 1/256`), the next rarest two. At grid point 0 (`n = 4000`) some cell is missing with
/// probability at most `sum_cells (1 - p)^4000 = 1.7e-7`.
const MED_PAG_BASE_N: usize = 8_000;

/// The enumerated natural direct effect of the `markov_parents` law: `5/32`.
const MED_NATURAL_DIRECT: f64 = 0.156_25;

/// The `mediation_mixed_graphs` fixture.
fn med_fixture() -> &'static Value {
    static FIXTURE: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    FIXTURE.get_or_init(|| {
        serde_json::from_str(include_str!(
            "../../../conformance/estimate/mediation_mixed_graphs/expected.json"
        ))
        .unwrap()
    })
}

/// Measure the natural direct effect on the one-atom PAG posterior of `markov_parents`.
fn med_pag_posterior(test: &'static str, family: u64, mode: Mode) {
    let case = &med_fixture()["markov_parents"];
    let truth = case["truth"]["natural_direct_effect"].as_f64().unwrap();
    assert_eq!(truth, MED_NATURAL_DIRECT, "the enumerated natural direct effect");
    let population = Population::from_contingency(case);
    assert_eq!(population.size(), 256, "the enumerated table");
    let nodes = u32::try_from(population.names.len()).unwrap();
    let edges: Vec<(u32, u32)> = case["graph"]["directed_edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| {
            (
                population.index(edge[0].as_str().unwrap()),
                population.index(edge[1].as_str().unwrap()),
            )
        })
        .collect();
    let pag = pag_of(nodes, &edges);
    let query = antecedent_core::CausalQuery::Mediation(antecedent_core::MediationQuery::binary(
        v(population.index("t")),
        v(population.index("y")),
        vec![v(population.index("m"))],
        antecedent_core::MediationContrast::NaturalDirect,
    ));
    measure_coordinate(
        &Coordinate {
            test,
            dgp: "conformance/estimate/mediation_mixed_graphs::markov_parents",
            family,
        },
        mode,
        truth,
        |seed, suite| {
            Study::tabular(population.sample(common::calibration::grid_n(MED_PAG_BASE_N), seed))
                .query(query.clone())
                .inference(mode.inference())
                .refute(suite)
                .graph_posterior(pag_posterior(&pag))
                .build()
                .unwrap_or_else(|e| panic!("{test} {suite:?}: {e}"))
        },
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn mediation_effect_pag_graph_posterior_frequentist_coverage() {
    med_pag_posterior(
        "mediation_effect_pag_graph_posterior_frequentist_coverage",
        0xFE70_0000,
        Mode::Frequentist,
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn mediation_effect_pag_graph_posterior_bayesian_coverage() {
    med_pag_posterior(
        "mediation_effect_pag_graph_posterior_bayesian_coverage",
        0xFE70_0001,
        Mode::Bayesian,
    );
}

// ------------------------------------------------ nested counterfactual on a PAG
//
// NestedCounterfactualEffect: the natural direct effect of `t` on `y` avoiding the mediator `m`
// in the `mediated_two_parents` table of `conformance/estimate/cross_world_class_truth`, on the
// PAG `c -> t`, `w -> m`, `t -> m`, `t -> y`, `m -> y` over the columns `c, w, t, m, y`, held as
// a one-atom graph posterior. By the mediation formula
// `sum_m P(m | t=0) [P(y=1 | t=1, m) - P(y=1 | t=0, m)] = 0.8 (0.5 - 0.1) + 0.2 (0.9 - 0.6)
// = 0.38`, which every candidate functional of the class takes on the table.

/// Base sample size of the 32-cell `mediated_two_parents` table: its smallest cell has
/// probability `6/2500 = 0.0024`, so at the grid's first point (`n = 7000`) the chance that any
/// cell is absent is at most `32 * 0.9976^7000 < 1e-6`.
const CW_WIDE_N: usize = 14_000;

/// The cross-world class fixture: the count tables and each case's exact functional.
fn cw_class_pin() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/cross_world_class_truth/expected.json"
    ))
    .unwrap()
}

/// The population of the fixture table `name`: `columns` and a list of cells, each with its
/// `levels` (one per column) and its `count`.
fn cw_population(pin: &Value, name: &str) -> Population {
    let spec = &pin["tables"][name];
    let names: Vec<&str> =
        spec["columns"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
    let cells = spec["counts"].as_array().unwrap().iter().map(|cell| {
        let levels: Vec<f64> =
            cell["levels"].as_array().unwrap().iter().map(|l| l.as_f64().unwrap()).collect();
        (levels, cell["count"].as_u64().unwrap())
    });
    Population::from_cells(&names, cells)
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn nested_counterfactual_effect_pag_graph_posterior_bayesian_coverage() {
    let test = "nested_counterfactual_effect_pag_graph_posterior_bayesian_coverage";
    let pin = cw_class_pin();
    let population = cw_population(&pin, "mediated_two_parents");
    let truth = pin["cases"]["pag_natural_direct"]["expected"].as_f64().unwrap();
    let pag = pag_of(5, &[(0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]);
    let query = antecedent_core::CausalQuery::NestedCounterfactual(
        antecedent_core::NestedCounterfactualQuery::new(v(2), v(3), v(4)).unwrap(),
    );
    measure_coordinate(
        &Coordinate { test, dgp: "cross_world_mediated_two_parents_pag_atom", family: 0xFE70_0002 },
        Mode::Bayesian,
        truth,
        |seed, suite| {
            Study::tabular(population.sample(common::calibration::grid_n(CW_WIDE_N), seed))
                .query(query.clone())
                .inference(Mode::Bayesian.inference())
                .refute(suite)
                .graph_posterior(pag_posterior(&pag))
                .build()
                .unwrap_or_else(|e| panic!("{test} {suite:?}: {e}"))
        },
    );
}

// ------------------------------------------------------ temporal window on a PAG
//
// TemporalMediationEffect: the pure natural indirect effect `E[Y(0, M(1))] - E[Y(0, M(0))]` of
// `x` on `y` through `m` at the pulse period, on the unrolled window of the stationary
// mediation template of `conformance/estimate/temporal_window_truth`, held as a one-atom graph
// posterior of a lagged `Pag` atom. Every edge into `m` and `y` and out of `x` is visible
// (`c(t-1)` points into `x`, `w` into `m`), and every candidate functional takes the natural
// indirect effect `1/8` on the stationary law.
//
// The rows of a window table are overlapping windows of one dependent series, so a replicate
// simulates the structural process itself, started in its stationary law.

/// Base series length of the mediation decomposition; `grid_n` measures `n/2, n, 2n`.
///
/// The edge g-formula reads `P(w)`, `P(m | x, w)` and `P(y | x, m)` at the pulse period: the
/// rarest conditioning cell `(x, m) = (0, 1)` has probability `3/16`, so every cell is present
/// at the smallest grid point and in every half-sample refit.
const TW_MEDIATION_BASE_N: usize = 1_000;

/// `q(x, w)` of the mediation process, indexed `[x][w]`: `m = 1` when a draw uniform on four
/// bins falls below it.
const TW_MEDIATOR_THRESHOLD: [[usize; 2]; 2] = [[1, 2], [2, 3]];

/// `s(x, m)` of the mediation process, indexed `[x][m]`: `y = 1` when a draw uniform on two
/// bins falls below it.
const TW_OUTCOME_THRESHOLD: [[usize; 2]; 2] = [[0, 1], [1, 1]];

/// The exact enumerated natural indirect effect of the mediation process: `1/8`.
const TW_NATURAL_INDIRECT: f64 = 0.125;

/// The mediation process's natural indirect effect, from the fixture, checked against its
/// closed form `(P(M(1) = 1) - P(M(0) = 1)) / 2 = (5/8 - 3/8) / 2`.
fn tw_truth() -> f64 {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_window_truth/expected.json"
    ))
    .unwrap();
    let value = fixture["mediation"]["truth"]["natural_indirect_effect"].as_f64().unwrap();
    assert!((value - TW_NATURAL_INDIRECT).abs() < 1e-15, "fixture truth {value}");
    value
}

/// `n` periods of `c, x, w, m, y` from the mediation process: `c`, `w` fair coins,
/// `x(t) = 1[Ux < 1 + 2 c(t-1)]` (`Ux` on four bins), `m = 1[Um < q(x, w)]` (`Um` on four
/// bins), `y = 1[Uy < s(x, m)]` (`Uy` on two bins).
///
/// `c` is iid, so drawing the period before the series as a fair coin starts the process in
/// its stationary law.
fn tw_mediation_series(n: usize, seed: u64) -> antecedent_data::TimeSeriesData {
    let mut u01 = uniform(seed, 0xFE70_3ED1);
    let mut bin = |bins: usize| ((u01() * bins as f64) as usize).min(bins - 1);
    let mut c_prev = bin(2);
    let mut columns: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::with_capacity(n));
    for _ in 0..n {
        let (ct, ux, wt, um, uy) = (bin(2), bin(4), bin(2), bin(4), bin(2));
        let xt = usize::from(ux < 1 + 2 * c_prev);
        let mt = usize::from(um < TW_MEDIATOR_THRESHOLD[xt][wt]);
        let yt = usize::from(uy < TW_OUTCOME_THRESHOLD[xt][mt]);
        for (column, value) in columns.iter_mut().zip([ct, xt, wt, mt, yt]) {
            column.push(value as f64);
        }
        c_prev = ct;
    }
    antecedent_data::TimeSeriesData::from_f64_columns(
        ["c", "x", "w", "m", "y"].into_iter().zip(columns.iter().map(Vec::as_slice)),
        1,
    )
    .unwrap()
}

/// The natural indirect effect of `x` (column 1) on `y` (column 4) through `m` (column 3).
fn tw_query() -> antecedent_core::CausalQuery {
    antecedent_core::CausalQuery::Mediation(antecedent_core::MediationQuery::binary(
        v(1),
        v(4),
        [v(3)],
        antecedent_core::MediationContrast::NaturalIndirect,
    ))
}

/// Measure the natural indirect effect on the one-atom lagged PAG posterior of the template
/// (`c = 0`, `x = 1`, `w = 2`, `m = 3`, `y = 4`): `c(t-1) -> x`, `x -> m`, `w -> m`, `m -> y`,
/// `x -> y`.
fn tw_pag_posterior(test: &'static str, family: u64, mode: Mode) {
    let query = tw_query();
    measure_coordinate(
        &Coordinate { test, dgp: "temporal_window_mediation_process", family },
        mode,
        tw_truth(),
        |seed, suite| {
            Study::series(tw_mediation_series(
                common::calibration::grid_n(TW_MEDIATION_BASE_N),
                seed,
            ))
            .graph_posterior(lagged_pag_posterior(
                5,
                &[(1, 3), (2, 3), (3, 4), (1, 4)],
                &[(1, 0, 1)],
            ))
            .query(query.clone())
            .inference(mode.inference())
            .refute(suite)
            .build()
            .unwrap_or_else(|e| panic!("{test} {suite:?}: {e}"))
        },
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_mediation_effect_temporal_pag_graph_posterior_frequentist_coverage() {
    tw_pag_posterior(
        "temporal_mediation_effect_temporal_pag_graph_posterior_frequentist_coverage",
        0xFE70_0003,
        Mode::Frequentist,
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_mediation_effect_temporal_pag_graph_posterior_bayesian_coverage() {
    tw_pag_posterior(
        "temporal_mediation_effect_temporal_pag_graph_posterior_bayesian_coverage",
        0xFE70_0004,
        Mode::Bayesian,
    );
}

//! Repeated-sampling coverage of the general identification functional estimator
//! (`functional.effect`) on every query, graph class, structure route and validation
//! mode it is offered under.
//!
//! One test measures one coordinate: a query, a graph class, an inference mode and a
//! structure family (a given graph, or a graph posterior). The validation modes and, for
//! a given graph, its accepted twin are sibling executions of that coordinate: the
//! driver measures the first listed execution on every replicate and requires every
//! sibling to report its interval bit for bit.
//!
//! Every discrete law is an exact finite population: the enumerated contingency table
//! of a structural model whose noises are equally likely bins (the tables of the
//! `conformance/estimate` known-truth fixtures). A replicate draws its rows independently
//! and uniformly from that table, so it samples the structural law itself, and the
//! population value of each identified functional is the fixture's exact truth.
//!
//! A graph posterior holds one atom: one identified atom publishes its own sampling
//! interval, whereas a mixture over atoms is a distribution over graphs.
//!
//! The driver scores the reported 0.95 interval and the 0.90 interval of the same
//! execution. A Frequentist interval is `estimate ± z·se` from the bootstrap standard
//! error the estimator publishes, re-formed at 0.90 from the same SE; a Bayesian
//! interval is the equal-tailed quantile interval of the retained posterior draws,
//! re-summarized at 0.90 from those draws. A class whose candidate structures identify the
//! effect by different functionals publishes only its identified set, at one level; those
//! coordinates score that interval at its own level (see `measure_identified_set`).
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

use antecedent::{
    AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study, StudyBuilder, StudyResult,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::adjacency_mask::{
    AdjacencyMask, adjacency_mask_from_admg, adjacency_mask_from_cpdag, adjacency_masks_from_pag,
    set_edge,
};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Pag, TemporalAdmg, TemporalPag};
use antecedent_prob::InferenceDiagnostics;
use common::calibration::{
    Construction, CoverageSide, CoverageTally, RecordKey, ScopeFacts, map_replicates, n_sim,
    stream_seed, uniform,
};
use common::calibration_bind::constructions;
use common::reported::{
    GATE_LEVEL, REPORTED_LEVEL, posterior_pair, scalar_normal_pair, scalar_reported_se,
};
use grid_cell::{
    Cell, IntervalPair, SIBLING_REPLICATES, Target, measure, measure_from_unrecorded_base,
    sharing_base,
};
use serde_json::Value;

/// Posterior draws of a Bayesian execution.
const DRAWS: usize = 400;

/// Estimator every coordinate of this file executes, unless a design names another
/// estimator its cell lists.
const FUNCTIONAL: &str = "functional.effect";

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
        Self::from_table(&case["columns"], &case["contingency_table"])
    }

    /// The population of a `columns` list and a list of cells, each an object with one
    /// value per column and a `count`.
    fn from_table(columns: &Value, table: &Value) -> Self {
        let names: Vec<&str> =
            columns.as_array().unwrap().iter().map(|name| name.as_str().unwrap()).collect();
        let cells = table.as_array().unwrap().iter().map(|cell| {
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

/// How an execution is given its structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Via {
    Explicit,
    Accepted,
    Posterior,
}

use Via::{Accepted, Explicit, Posterior};

/// One execution of a coordinate: its structure route and validation mode.
type Variant = (Via, RefuteSuite);

fn directed_mask(nodes: usize, edges: &[(u32, u32)]) -> AdjacencyMask {
    edges.iter().fold(AdjacencyMask::EMPTY, |mask, &(from, to)| {
        set_edge(mask, nodes, from as usize, to as usize, true)
    })
}

/// A graph posterior holding all its mass on one atom.
fn one_atom(
    nodes: usize,
    atom: AdjacencyMask,
    kind: GraphPosteriorAtomKind,
    marks: Option<AdjacencyMask>,
) -> GraphPosterior {
    let posterior = GraphPosterior::from_atoms(
        nodes,
        vec![1.0],
        vec![atom],
        vec![0.0; nodes * nodes],
        vec![0.0; nodes * nodes],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("single_atom_functional"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(kind);
    match marks {
        Some(marks) => posterior.with_mark_atoms(vec![marks]).unwrap(),
        None => posterior,
    }
}

/// The structure of a design, given as a graph or as a one-atom graph posterior.
enum Structure {
    Dag {
        nodes: u32,
        edges: Vec<(u32, u32)>,
    },
    Cpdag(Cpdag),
    Pag(Pag),
    Admg(Admg),
    /// The contemporaneous `TemporalAdmg` of a static mixed graph (every edge at lag 0);
    /// its posterior atom is the `TemporalAdmg` kind with the same packing.
    ContemporaneousAdmg(Admg),
}

impl Structure {
    fn dag(nodes: u32, edges: &[(u32, u32)]) -> Dag {
        let mut graph = Dag::with_variables(nodes);
        for &(from, to) in edges {
            graph.insert_directed(d(from), d(to)).unwrap();
        }
        graph
    }

    fn posterior(&self) -> GraphPosterior {
        match self {
            Self::Dag { nodes, edges } => {
                let nodes = *nodes as usize;
                one_atom(nodes, directed_mask(nodes, edges), GraphPosteriorAtomKind::Dag, None)
            }
            Self::Cpdag(graph) => one_atom(
                graph.node_count(),
                adjacency_mask_from_cpdag(graph).unwrap(),
                GraphPosteriorAtomKind::Cpdag,
                None,
            ),
            Self::Pag(graph) => {
                let (adjacency, marks) = adjacency_masks_from_pag(graph).unwrap();
                one_atom(graph.node_count(), adjacency, GraphPosteriorAtomKind::Pag, Some(marks))
            }
            Self::Admg(graph) => one_atom(
                graph.node_count(),
                adjacency_mask_from_admg(graph).unwrap(),
                GraphPosteriorAtomKind::Admg,
                None,
            ),
            Self::ContemporaneousAdmg(graph) => one_atom(
                graph.node_count(),
                adjacency_mask_from_admg(graph).unwrap(),
                GraphPosteriorAtomKind::TemporalAdmg,
                None,
            ),
        }
    }

    fn attach(&self, builder: StudyBuilder, via: Via) -> StudyBuilder {
        match (self, via) {
            (_, Posterior) => builder.graph_posterior(self.posterior()),
            (Self::Dag { nodes, edges }, Explicit) => builder.graph(Self::dag(*nodes, edges)),
            (Self::Dag { nodes, edges }, Accepted) => {
                builder.graph(AcceptedGraph::from(Self::dag(*nodes, edges)))
            }
            (Self::Cpdag(graph), Explicit) => builder.graph(graph.clone()),
            (Self::Cpdag(graph), Accepted) => builder.graph(AcceptedGraph::from(graph.clone())),
            (Self::Pag(graph), Explicit) => builder.graph(graph.clone()),
            (Self::Pag(graph), Accepted) => builder.graph(AcceptedGraph::from(graph.clone())),
            (Self::Admg(graph), Explicit) => builder.graph(graph.clone()),
            (Self::Admg(graph), Accepted) => builder.graph(AcceptedGraph::from(graph.clone())),
            (Self::ContemporaneousAdmg(graph), Explicit) => {
                builder.graph(TemporalAdmg::from_static_contemporaneous(graph).unwrap())
            }
            (Self::ContemporaneousAdmg(graph), Accepted) => builder.graph(AcceptedGraph::from(
                TemporalAdmg::from_static_contemporaneous(graph).unwrap(),
            )),
        }
    }
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

    /// `IntervalMethod::as_str()` of the scalar interval the mode reports: the normal
    /// interval from the bootstrap SE, or the quantiles of the posterior draws.
    fn interval(self) -> &'static str {
        match self {
            Self::Frequentist => "bootstrap_se",
            Self::Bayesian => "posterior_quantile",
        }
    }
}

/// `[reported, gate]` interval of a scalar effect: the bootstrap-SE normal interval of
/// a Frequentist execution, the posterior effect column of a Bayesian one.
fn effect_pair(mode: Mode, result: &StudyResult) -> IntervalPair {
    match mode {
        Mode::Frequentist => {
            assert_eq!(scalar_reported_se(result).1, "bootstrap_se", "bootstrap SE reported");
            scalar_normal_pair(result)
        }
        Mode::Bayesian => {
            let column = result
                .posterior
                .as_ref()
                .and_then(antecedent::CausalPosterior::effect_column)
                .expect("a posterior effect column");
            posterior_pair(result, column)
        }
    }
}

// ------------------------------------------------------------------- execution

/// The coordinate one test measures.
struct Coordinate<'a> {
    /// The test's own name.
    test: &'static str,
    /// The generator's name.
    dgp: &'static str,
    /// Seed family of the test's replicates.
    family: u64,
    /// The estimator the execution must report.
    estimator: &'static str,
    /// `IntervalMethod::as_str()` of the scored interval.
    interval: &'static str,
    /// `variants[0]` is measured on every replicate, the rest are its siblings.
    variants: &'a [Variant],
}

/// Measure `coordinate`: `study(seed, variant)` builds one execution on the replicate's
/// dataset (which must depend on `seed` only); `score` reads one interval pair per
/// target, in `targets` order.
fn measure_coordinate(
    coordinate: &Coordinate<'_>,
    targets: &[Target],
    study: impl Fn(u64, Variant) -> Study + Sync,
    score: impl Fn(&StudyResult) -> Vec<IntervalPair> + Sync,
) {
    measure_coordinate_from(coordinate, targets, study, score, BaseRecord::Recorded);
}

/// Whether a coordinate's base execution (`variants[0]`) backs a record of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BaseRecord {
    Recorded,
    /// The base is a validation-none execution of a cell another test records; only the
    /// listed siblings are recorded (`grid_cell::measure_from_unrecorded_base`).
    Unrecorded,
}

/// [`measure_coordinate`] with the base execution recorded or not.
fn measure_coordinate_from(
    coordinate: &Coordinate<'_>,
    targets: &[Target],
    study: impl Fn(u64, Variant) -> Study + Sync,
    score: impl Fn(&StudyResult) -> Vec<IntervalPair> + Sync,
    base: BaseRecord,
) {
    let cell = Cell {
        test: coordinate.test,
        dgp: coordinate.dgp,
        interval: coordinate.interval,
        estimator: coordinate.estimator,
        variants: coordinate.variants.len(),
    };
    let run = |rep: u64, variant: usize| {
        let seed = stream_seed(coordinate.family, rep);
        let execution = coordinate.variants[variant];
        let study = study(seed, execution);
        let result = study
            .run(&ExecutionContext::for_tests(seed))
            .unwrap_or_else(|e| panic!("{} {execution:?}: {e}", coordinate.test));
        (study, result)
    };
    match base {
        BaseRecord::Recorded => measure(cell, targets, run, score),
        BaseRecord::Unrecorded => measure_from_unrecorded_base(cell, targets, run, score),
    }
}

// ------------------------------------------------------------ identified sets
//
// A class whose candidate structures identify the effect by different functionals (the ADMGs a
// PAG's MAG completion stands for, `execute_pag_path_specific`) publishes the
// enumeration-weighted mean with no standard error and no posterior, and its one interval is
// the class identified set (`structural_response.identified_set_interval`, bound as
// `identified_set`): a Frequentist execution the Imbens-Manski interval from every candidate
// refitted on shared bootstrap resamples, at 0.95 (`class_path_intervals`); a Bayesian one the
// product-posterior envelope of the candidates' Bayesian-bootstrap posteriors, at 0.90
// (`IDENTIFIED_SET_INTERVAL_LEVEL`). Each is published at that one level only, so these
// coordinates score the one published interval at its own level.
//
// Every candidate functional evaluates to the true effect on the population law (the
// generating DAG is Markov to every candidate the class admits, and the confounding a
// candidate adds touches only a root's edge into the treatment or the mediator), so the
// population identified set is the point at the truth, and the sample set's spread is
// sampling noise only. The interval is then conservative by construction (it widens to cover
// every candidate's estimate), so each record is gated one-sided: coverage at least the level,
// with no upper band (`CoverageTally::assert_one_sided`).

/// `IntervalMethod::as_str()` of a class identified-set interval.
const IDENTIFIED_SET: &str = "identified_set";

/// Level a class identified set is published at under `mode`.
fn identified_set_level(mode: Mode) -> f64 {
    match mode {
        Mode::Frequentist => 0.95,
        Mode::Bayesian => 0.9,
    }
}

/// The identified-set interval of `result`, checked to be published at `level`; `None` when
/// the candidates agreed on this sample and the result published a point.
fn identified_set(result: &StudyResult, level: f64) -> Option<(f64, f64)> {
    let set = result.structural_response.as_ref()?.identified_set_interval.as_ref()?;
    assert!((set.level - level).abs() < 1e-12, "identified set published at {}", set.level);
    Some((set.lower, set.upper))
}

/// An identified-set coordinate: the test, its generator, seed family, mode, executions and the
/// one-sided target its records state.
struct SetCoordinate<'a> {
    test: &'static str,
    dgp: &'static str,
    family: u64,
    mode: Mode,
    /// `variants[0]` is measured on every replicate, the rest are its siblings.
    variants: &'a [Variant],
    /// What the one-sided record covers.
    target: &'static str,
}

/// One execution's identified set and the runtime's binding of it.
type SetRun = (Option<(f64, f64)>, Option<(Construction, ScopeFacts, &'static str)>);

/// Measure the identified set of `coordinate` against `truth` over `n_sim()` replicates.
///
/// The driver of `common/grid_cell.rs` at one level: `variants[0]` is scored on every
/// replicate; every sibling is executed on the first [`SIBLING_REPLICATES`] replicates, must
/// publish the base interval bit for bit, and gets its own record bound from its own
/// executions' match key and validation mode. A replicate whose candidates agreed publishes no
/// set and is scored as a miss.
fn measure_identified_set(
    coordinate: &SetCoordinate<'_>,
    truth: f64,
    study: impl Fn(u64, Variant) -> Study + Sync,
) {
    let level = identified_set_level(coordinate.mode);
    let test = coordinate.test;
    let execute = |rep: u64, variant: usize| -> SetRun {
        let seed = stream_seed(coordinate.family, rep);
        let execution = coordinate.variants[variant];
        let study = study(seed, execution);
        let result = study
            .run(&ExecutionContext::for_tests(seed))
            .unwrap_or_else(|e| panic!("{test} {execution:?}: {e}"));
        assert_eq!(
            result.logical_plan.estimator.as_deref(),
            Some(FUNCTIONAL),
            "{test} {execution:?}: estimator target"
        );
        let interval = identified_set(&result, level);
        let bound = constructions(&study.inspect().expect("inspect"), &result)
            .into_iter()
            .find(|(construction, _, _)| construction.interval_method == IDENTIFIED_SET);
        assert_eq!(
            interval.is_some(),
            bound.is_some(),
            "{test} {execution:?}: the runtime binds the identified set exactly when it is \
             published"
        );
        (interval, bound)
    };
    let bits = |interval: Option<(f64, f64)>| interval.map(|(lo, hi)| (lo.to_bits(), hi.to_bits()));
    let runs = map_replicates(n_sim(), |rep| {
        let base = execute(rep, 0);
        let siblings: Vec<SetRun> = if rep < u64::from(SIBLING_REPLICATES) {
            (1..coordinate.variants.len())
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
    let keys: Vec<(Construction, &'static str)> = (1..coordinate.variants.len())
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

    let tallies: Vec<CoverageTally> = (0..coordinate.variants.len())
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
                tally.assert_one_sided(CoverageSide::Lower, coordinate.target);
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

// ---------------------------------------------------------- path-specific effect
//
// PathSpecificEffect: the effect of `t` on `y` along the paths through `a` only,
// `E[Y(A(1), B(0))] - E[Y(A(0), B(0))]` (the treatment enters `a` at the active level and
// every other child at the control level), on a supplied Dag, Admg, the contemporaneous
// projection of a TemporalAdmg, Cpdag and Pag, and on a one-atom graph posterior of each.
//
// The given-structure and Markovian posterior laws are the binary structural models of
// `conformance/estimate/path_specific_mixed_graphs`: every noise is uniform on four
// equally likely bins, and the fixture's contingency table is the exact observed law
// (one row per noise cell, counts divided by their common factor). A replicate draws its
// rows uniformly from that table, so it samples the structural law itself.
//
// A graph posterior stores an atom as an adjacency mask, which cannot hold the directed and
// the bidirected edge of one pair (`adjacency_mask_from_admg` refuses a bow), so the
// fixture's latent-confounded graph (`a -> y` with `a <-> y`) has no posterior atom. The ADMG
// and TemporalAdmg posterior coordinates measure [`ps_confounded_treatment`] instead: a law
// whose latent confounder joins the treatment and the outcome, a pair no directed edge
// joins, so its graph is bow-free and still identified by the district formula.

/// Known-truth fixture of the path-specific laws on mixed graphs and classes.
fn ps_fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/path_specific_mixed_graphs/expected.json"
    ))
    .unwrap()
}

/// One path-specific design: the law, the structure of the coordinate's graph class, the
/// query and the exact population truth.
struct PsDesign {
    dgp: &'static str,
    population: Population,
    structure: Structure,
    query: antecedent_core::CausalQuery,
    truth: f64,
    /// Base sample size: at grid point 0 (`base_n / 2` rows) every positive-mass cell of
    /// the population is present with probability above `1 - 1e-6`.
    base_n: usize,
}

/// `district_admg` rows: 256, the rarest cells `(t=0, a=1, b=1, y=0)` and
/// `(t=1, a=0, b=0, y=1)` hold one row each (`p = 1/256`), the next rarest three. At grid
/// point 0 (`n = 4000`) a cell is missing with probability at most
/// `2 (1 - 1/256)^4000 + (1 - 3/256)^4000 ≈ 3.3e-7`.
const PS_DISTRICT_BASE_N: usize = 8_000;

/// `routed_markov` rows: 512, the rarest cell `(w=0, t=1, a=0, b=0, y=1)` holds one row
/// (`p = 1/512`), the next rarest three. At grid point 0 (`n = 7500`) a cell is missing with
/// probability at most `(1 - 1/512)^7500 + 2 (1 - 3/512)^7500 ≈ 4.3e-7`.
const PS_ROUTED_BASE_N: usize = 15_000;

/// Base sample size of [`ps_confounded_treatment`]: 512 rows, the rarest cell
/// `(t=0, a=1, b=1, y=0)` holds three (`p = 3/512`), the next rarest seven. At grid point 0
/// (`n = 2500`) some cell is missing with probability at most `Σ_cells (1 - p)^2500 ≈ 4e-7`.
const PS_CONFOUNDED_BASE_N: usize = 5_000;

/// The bow-free latent-confounded path-specific law of the ADMG and TemporalAdmg posterior
/// coordinates, enumerated exactly: latent `U ~ Bern(1/2)`, `T = 1[Ut < (1 + 2U)/4]`,
/// `A = 1[Ua < (1 + 2T)/4]`, `B = 1[Ub < (1 + 2T)/4]`, `Y = 1[Uy < (1 + A + B + U)/4]`, every
/// noise uniform on four bins (`U` on two). One population row per noise cell (512 rows over
/// the columns `t, a, b, y`); the graph is `t -> a, t -> b, a -> y, b -> y, t <-> y`.
///
/// `P(Y = 1 | a, b, u) = (1 + a + b + u)/4` is linear in `a` and `A(t)` does not depend on
/// `U`, so `E[Y(A(1), B(0))] - E[Y(A(0), B(0))] = E[A(1) - A(0)]/4 = (3/4 - 1/4)/4 = 0.125`.
/// The truth returned is that cross-world contrast summed over the same noise cells (exact in
/// binary floating point), checked against the closed form. The identified functional
/// `Σ_{a,b} [P(a | t=1) - P(a | t=0)] P(b | t=0) Σ_t' P(y=1 | a, b, t') P(t')` (the district
/// formula: `t` blocks the back-door `a <- t <-> y`) takes the same value on the table.
fn ps_confounded_treatment() -> (Population, f64) {
    let mut cells = Vec::with_capacity(512);
    let mut contrast = 0.0;
    for u in 0..2_usize {
        for ut in 0..4 {
            for ua in 0..4 {
                for ub in 0..4 {
                    for uy in 0..4 {
                        let a_at = |t: usize| usize::from(ua < 1 + 2 * t);
                        let b_at = |t: usize| usize::from(ub < 1 + 2 * t);
                        let y_at = |a: usize, b: usize| usize::from(uy < 1 + a + b + u);
                        let t = usize::from(ut < 1 + 2 * u);
                        let (a, b) = (a_at(t), b_at(t));
                        cells.push((vec![t as f64, a as f64, b as f64, y_at(a, b) as f64], 1));
                        contrast += y_at(a_at(1), b_at(0)) as f64 - y_at(a_at(0), b_at(0)) as f64;
                    }
                }
            }
        }
    }
    let population = Population::from_cells(&["t", "a", "b", "y"], cells);
    let truth = contrast / population.size() as f64;
    assert_eq!(truth, 0.125, "the enumerated path-specific effect");
    (population, truth)
}

/// The design of graph class `class`, given (`posterior == false`) or as a one-atom graph
/// posterior.
///
/// * `district_admg` (Admg and the contemporaneous TemporalAdmg): latent `U ~ Bern(1/2)`,
///   `T ~ Bern(1/2)`, `A = 1[Ua < (1 + 2T + U)/4]`, `B = 1[Ub < (1 + 2T)/4]`,
///   `Y = 1[Uy < (1 + A + B + U)/4]`; graph `t -> a, t -> b, a -> y, b -> y, a <-> y`.
///   `P(Y = 1 | a, b, u) = (1 + a + b + u)/4` is linear in `a`, so the effect is
///   `E[A(1, U) - A(0, U)]/4 = (2/4)/4 = 0.125`. The district formula
///   `sum_b P(b | t=0) (E[Y | t=1, b] - E[Y | t=0, b])` gives the same on the table:
///   `0.75 (19/32 - 45/96) + 0.25 (81/96 - 23/32) = 0.75·0.125 + 0.25·0.125 = 0.125`.
/// * `routed_markov` (Dag, Cpdag with every edge compelled, Pag with every mark decided):
///   `W ~ Bern(1/2)`, `T = 1[Ut < (1 + 2W)/4]`, `A = 1[Ua < (1 + 2T)/4]`,
///   `B = 1[Ub < (1 + 2T)/4]`, `P(Y = 1 | a, b) = (1, 2, 2, 4)/4` for
///   `(a, b) = (0,0), (1,0), (0,1), (1,1)`; graph `w -> t, t -> a, t -> b, a -> y, b -> y`.
///   With `B(0) ~ Bern(1/4)` and `A(t) ~ Bern(p)`, `E[Y] = 0.75 (0.25 + 0.25p) +
///   0.25 (0.5 + 0.5p) = 0.3125 + 0.3125p`, so the effect is `0.3125 (3/4 - 1/4) = 0.15625`
///   (the edge g-formula `sum_{a,b} P(b | t=0) (P(a | t=1) - P(a | t=0)) E[Y | a, b]`).
///
/// The truths are read from the fixture, whose generator derives them from the structural
/// equations with exact rational arithmetic and checks the identified functional on the
/// table reproduces them.
fn ps_design(class: &str, posterior: bool) -> PsDesign {
    if posterior && matches!(class, "Admg" | "TemporalAdmg") {
        let (population, truth) = ps_confounded_treatment();
        let mut graph = Admg::with_variables(4);
        for (from, to) in [(0, 1), (0, 2), (1, 3), (2, 3)] {
            graph.insert_directed(d(from), d(to)).unwrap();
        }
        graph.insert_bidirected(d(0), d(3)).unwrap();
        return PsDesign {
            dgp: "confounded_treatment",
            population,
            structure: if class == "Admg" {
                Structure::Admg(graph)
            } else {
                Structure::ContemporaneousAdmg(graph)
            },
            query: antecedent_core::CausalQuery::PathSpecific(
                antecedent_core::PathSpecificEffectQuery::binary(v(0), v(3))
                    .with_path_nodes(vec![v(1)])
                    .with_max_len(4),
            ),
            truth,
            base_n: PS_CONFOUNDED_BASE_N,
        };
    }
    let fixture = ps_fixture();
    let (key, base_n) = match class {
        "Admg" | "TemporalAdmg" => ("district_admg", PS_DISTRICT_BASE_N),
        "Dag" | "Cpdag" | "Pag" => ("routed_markov", PS_ROUTED_BASE_N),
        other => panic!("no path-specific design for {other}"),
    };
    let case = &fixture[key];
    let population = Population::from_contingency(case);
    let node = |name: &str| population.index(name);
    let edges = |list: &str| -> Vec<(u32, u32)> {
        case["graph"][list]
            .as_array()
            .map(|edges| {
                edges
                    .iter()
                    .map(|edge| (node(edge[0].as_str().unwrap()), node(edge[1].as_str().unwrap())))
                    .collect()
            })
            .unwrap_or_default()
    };
    let directed = edges("directed_edges");
    let nodes = u32::try_from(population.names.len()).unwrap();
    let structure = match class {
        "Dag" => Structure::Dag { nodes, edges: directed },
        "Cpdag" => {
            // The Markovian graph with every edge oriented: a class with a single completion.
            let mut graph = Cpdag::with_variables(nodes);
            for &(from, to) in &directed {
                graph.insert_directed(d(from), d(to)).unwrap();
            }
            Structure::Cpdag(graph)
        }
        "Pag" => {
            // Every mark decided: the edges out of `t` and into `y` are visible because the
            // root `w` points into `t`.
            let mut graph = Pag::with_variables(nodes);
            for &(from, to) in &directed {
                graph.insert_directed(d(from), d(to)).unwrap();
            }
            Structure::Pag(graph)
        }
        _ => {
            let mut graph = Admg::with_variables(nodes);
            for &(from, to) in &directed {
                graph.insert_directed(d(from), d(to)).unwrap();
            }
            for (left, right) in edges("bidirected_edges") {
                graph.insert_bidirected(d(left), d(right)).unwrap();
            }
            if class == "Admg" {
                Structure::Admg(graph)
            } else {
                Structure::ContemporaneousAdmg(graph)
            }
        }
    };
    let spec = &case["query"];
    let path_nodes: Vec<VariableId> = spec["path_nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|name| v(node(name.as_str().unwrap())))
        .collect();
    let query = antecedent_core::CausalQuery::PathSpecific(
        antecedent_core::PathSpecificEffectQuery::binary(
            v(node(spec["treatment"].as_str().unwrap())),
            v(node(spec["outcome"].as_str().unwrap())),
        )
        .with_path_nodes(path_nodes)
        .with_max_len(4),
    );
    PsDesign {
        dgp: if key == "district_admg" { "district_admg" } else { "routed_markov" },
        truth: case["truth"]["path_specific_effect"].as_f64().unwrap(),
        population,
        structure,
        query,
        base_n,
    }
}

/// Given-structure variants: explicit and accepted under every validation mode.
const PS_GIVEN: [Variant; 6] = [
    (Explicit, RefuteSuite::None),
    (Accepted, RefuteSuite::None),
    (Explicit, RefuteSuite::Cheap),
    (Accepted, RefuteSuite::Cheap),
    (Explicit, RefuteSuite::Full),
    (Accepted, RefuteSuite::Full),
];

/// Given-Dag variants: the explicit Dag without validation is measured elsewhere
/// (`static_calibration.rs`), so the accepted twin without validation is the base here.
const PS_GIVEN_DAG: [Variant; 5] = [
    (Accepted, RefuteSuite::None),
    (Explicit, RefuteSuite::Cheap),
    (Accepted, RefuteSuite::Cheap),
    (Explicit, RefuteSuite::Full),
    (Accepted, RefuteSuite::Full),
];

/// Graph-posterior variants: a one-atom posterior under every validation mode.
const PS_POSTERIOR: [Variant; 3] = [
    (Posterior, RefuteSuite::None),
    (Posterior, RefuteSuite::Cheap),
    (Posterior, RefuteSuite::Full),
];

/// Measure the path-specific coordinate of `class` under `mode`.
///
/// The interval is the one the runtime binds for the result: the posterior quantiles of a
/// Bayesian execution; for a Frequentist one the normal interval from the bootstrap SE
/// (`bootstrap_se`), except on a TemporalAdmg structure (given, or a posterior of
/// TemporalAdmg atoms), whose contemporaneous-projection functional resamples circular
/// blocks of consecutive rows (`Study::functional_row_resampling` selects
/// `RowResampling::SeriesBlocks`, and `FunctionalEffect::estimate` marks the estimate with
/// the row-block family), so the runtime binds `circular_block_se` to `est ± z·se` with the
/// block-bootstrap SE.
fn ps_coverage(test: &'static str, family: u64, class: &str, mode: Mode, variants: &[Variant]) {
    let design = ps_design(class, variants[0].0 == Posterior);
    let interval = match (mode, class) {
        (Mode::Frequentist, "TemporalAdmg") => "circular_block_se",
        _ => mode.interval(),
    };
    let n = common::calibration::grid_n(design.base_n);
    let design = &design;
    measure_coordinate(
        &Coordinate { test, dgp: design.dgp, family, estimator: FUNCTIONAL, interval, variants },
        &Target::scalar(design.truth),
        |seed, (via, suite)| {
            let builder = Study::tabular(design.population.sample(n, seed))
                .query(design.query.clone())
                .inference(mode.inference())
                .refute(suite);
            design
                .structure
                .attach(builder, via)
                .build()
                .unwrap_or_else(|e| panic!("{test} {via:?} {suite:?}: {e}"))
        },
        |result| vec![effect_pair(mode, result)],
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_dag_given_frequentist_coverage() {
    sharing_base(|| {
        {
            ps_coverage(
                "path_specific_dag_given_frequentist_coverage",
                0xFE10_0000,
                "Dag",
                Mode::Frequentist,
                &PS_GIVEN_DAG,
            );
        }
        {
            ps_coverage(
                "path_specific_dag_given_frequentist_coverage",
                0xFE10_0000,
                "Cpdag",
                Mode::Frequentist,
                &PS_GIVEN,
            );
        }
        {
            ps_coverage(
                "path_specific_dag_given_frequentist_coverage",
                0xFE10_0000,
                "Pag",
                Mode::Frequentist,
                &PS_GIVEN,
            );
        }
        {
            ps_coverage(
                "path_specific_dag_given_frequentist_coverage",
                0xFE10_0000,
                "Dag",
                Mode::Frequentist,
                &PS_POSTERIOR,
            );
        }
        {
            ps_coverage(
                "path_specific_dag_given_frequentist_coverage",
                0xFE10_0000,
                "Cpdag",
                Mode::Frequentist,
                &PS_POSTERIOR,
            );
        }
        {
            ps_coverage(
                "path_specific_dag_given_frequentist_coverage",
                0xFE10_0000,
                "Pag",
                Mode::Frequentist,
                &PS_POSTERIOR,
            );
        }
    });
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_dag_given_bayesian_coverage() {
    sharing_base(|| {
        {
            ps_coverage(
                "path_specific_dag_given_bayesian_coverage",
                0xFE10_0001,
                "Dag",
                Mode::Bayesian,
                &PS_GIVEN_DAG,
            );
        }
        {
            ps_coverage(
                "path_specific_dag_given_bayesian_coverage",
                0xFE10_0001,
                "Cpdag",
                Mode::Bayesian,
                &PS_GIVEN,
            );
        }
        {
            ps_coverage(
                "path_specific_dag_given_bayesian_coverage",
                0xFE10_0001,
                "Pag",
                Mode::Bayesian,
                &PS_GIVEN,
            );
        }
        {
            ps_coverage(
                "path_specific_dag_given_bayesian_coverage",
                0xFE10_0001,
                "Dag",
                Mode::Bayesian,
                &PS_POSTERIOR,
            );
        }
        {
            ps_coverage(
                "path_specific_dag_given_bayesian_coverage",
                0xFE10_0001,
                "Cpdag",
                Mode::Bayesian,
                &PS_POSTERIOR,
            );
        }
        {
            ps_coverage(
                "path_specific_dag_given_bayesian_coverage",
                0xFE10_0001,
                "Pag",
                Mode::Bayesian,
                &PS_POSTERIOR,
            );
        }
    });
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_admg_given_frequentist_coverage() {
    ps_coverage(
        "path_specific_admg_given_frequentist_coverage",
        0xFE10_0002,
        "Admg",
        Mode::Frequentist,
        &PS_GIVEN,
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_admg_given_bayesian_coverage() {
    ps_coverage(
        "path_specific_admg_given_bayesian_coverage",
        0xFE10_0003,
        "Admg",
        Mode::Bayesian,
        &PS_GIVEN,
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_temporal_admg_given_frequentist_coverage() {
    ps_coverage(
        "path_specific_temporal_admg_given_frequentist_coverage",
        0xFE10_0004,
        "TemporalAdmg",
        Mode::Frequentist,
        &PS_GIVEN,
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_temporal_admg_given_bayesian_coverage() {
    ps_coverage(
        "path_specific_temporal_admg_given_bayesian_coverage",
        0xFE10_0005,
        "TemporalAdmg",
        Mode::Bayesian,
        &PS_GIVEN,
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_admg_graph_posterior_frequentist_coverage() {
    ps_coverage(
        "path_specific_admg_graph_posterior_frequentist_coverage",
        0xFE10_000C,
        "Admg",
        Mode::Frequentist,
        &PS_POSTERIOR,
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_admg_graph_posterior_bayesian_coverage() {
    ps_coverage(
        "path_specific_admg_graph_posterior_bayesian_coverage",
        0xFE10_000D,
        "Admg",
        Mode::Bayesian,
        &PS_POSTERIOR,
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_temporal_admg_graph_posterior_frequentist_coverage() {
    ps_coverage(
        "path_specific_temporal_admg_graph_posterior_frequentist_coverage",
        0xFE10_000E,
        "TemporalAdmg",
        Mode::Frequentist,
        &PS_POSTERIOR,
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_temporal_admg_graph_posterior_bayesian_coverage() {
    ps_coverage(
        "path_specific_temporal_admg_graph_posterior_bayesian_coverage",
        0xFE10_000F,
        "TemporalAdmg",
        Mode::Bayesian,
        &PS_POSTERIOR,
    );
}

// ---- MediationEffect: the natural direct effect of a mediation decomposition
//
// A mediation execution evaluates the total, natural direct and pure natural indirect
// effects on the same rows, but it publishes an interval for the requested contrast only:
// the other components are point values beside it (and, for paired draws, moments in the
// `estimate.mediation.joint` disclosure), and the calibration key names the requested
// contrast (`natural_direct.all_observed`). Each coordinate therefore requests the natural
// direct effect and scores its one reported interval.
//
// The law is the `covariate_latent` table of `conformance/estimate/mediation_mixed_graphs`:
// the exact observed law of a binary structural model with a latent `U` behind `W` and `Y`,
// every noise uniform on four equally likely bins,
//
//     U = 1[Uu < 2/4]      W = 1[Uw < (1 + 2U)/4]      T = 1[Ut < (1 + 2W)/4]
//     M = 1[Um < (1 + W + T)/4]      Y = 1[Uy < (1 + min(1, T + M) + min(1, U + W))/4]
//
// so the table (512 rows, 16 cells, the smallest cell 8 rows) is the population and its
// natural direct effect `E[Y(1, M(0))] - E[Y(0, M(0))] = 5/32 = 0.15625` is the enumerated
// structural truth. `U` is independent of `T` and `M` given `W`, so every structure below
// identifies the same covariate-adjusted functional
// `sum_{w,m} P(w) P(m | T=0, w) [E(Y | 1, w, m) - E(Y | 0, w, m)]`, whose value on the table
// is that truth (`reference.py` checks the identity in exact arithmetic).

/// Base sample size of every mediation design. Grid point 0 draws `n = 1100` rows; each of
/// the 16 population cells has mass at least `8/512 = 1/64`, so the chance that any cell
/// (hence any stratum the plug-in conditions on) is empty is at most
/// `16 · (63/64)^1100 ≈ 16 · 3.0e-8 ≈ 4.9e-7`.
const MED_BASE_N: usize = 2200;

/// The enumerated natural direct effect of the `covariate_latent` law: `5/32`.
const MED_NATURAL_DIRECT: f64 = 0.156_25;

/// The generator every mediation design samples.
const MED_DGP: &str = "conformance/estimate/mediation_mixed_graphs::covariate_latent";

/// The directed edges of the `covariate_latent` structure.
const MED_DIRECTED: [(&str, &str); 6] =
    [("w", "t"), ("w", "m"), ("w", "y"), ("t", "m"), ("t", "y"), ("m", "y")];

/// A given structure: explicit and accepted, under every validation mode.
const MED_GIVEN: &[Variant] = &[
    (Explicit, RefuteSuite::None),
    (Explicit, RefuteSuite::Cheap),
    (Explicit, RefuteSuite::Full),
    (Accepted, RefuteSuite::None),
    (Accepted, RefuteSuite::Cheap),
    (Accepted, RefuteSuite::Full),
];

/// A one-atom graph posterior, under every validation mode.
const MED_POSTERIOR: &[Variant] = &[
    (Posterior, RefuteSuite::None),
    (Posterior, RefuteSuite::Cheap),
    (Posterior, RefuteSuite::Full),
];

/// The graph class a mediation coordinate is executed on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MedClass {
    Admg,
    TemporalAdmg,
    Cpdag,
}

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

/// The structure of a coordinate over the population's columns.
///
/// * A given ADMG (and the contemporaneous `TemporalAdmg` of it) is the fixture's graph: the
///   six directed edges and the latent `W <-> Y`.
/// * A graph posterior stores an atom as an adjacency mask, which cannot hold the directed
///   and the bidirected edge of one pair, so its ADMG atom is the six directed edges alone
///   (the fixture `functional_graph_posterior_mixtures` atom `covariate_adjusted`). The
///   identified functional is the same covariate adjustment, so the truth is unchanged.
/// * A CPDAG (given, or the posterior atom `adjusted`) has the six edges oriented: a class
///   with one completion, the covariate-adjusted DAG.
fn med_structure(population: &Population, class: MedClass, posterior: bool) -> Structure {
    let nodes = u32::try_from(population.names.len()).unwrap();
    let node = |name: &str| d(population.index(name));
    match class {
        MedClass::Cpdag => {
            let mut cpdag = Cpdag::with_variables(nodes);
            for (from, to) in MED_DIRECTED {
                cpdag.insert_directed(node(from), node(to)).unwrap();
            }
            Structure::Cpdag(cpdag)
        }
        MedClass::Admg | MedClass::TemporalAdmg => {
            let mut admg = Admg::with_variables(nodes);
            for (from, to) in MED_DIRECTED {
                admg.insert_directed(node(from), node(to)).unwrap();
            }
            if !posterior {
                admg.insert_bidirected(node("w"), node("y")).unwrap();
            }
            if class == MedClass::Admg {
                Structure::Admg(admg)
            } else {
                Structure::ContemporaneousAdmg(admg)
            }
        }
    }
}

/// One execution: the natural direct effect of `t` on `y` through `m` on `grid_n(base_n)` rows
/// drawn from the population, with the study's default bootstrap.
fn med_study(
    (population, base_n): (&Population, usize),
    structure: &Structure,
    mode: Mode,
    seed: u64,
    (via, suite): Variant,
) -> Study {
    let data = population.sample(common::calibration::grid_n(base_n), seed);
    let query = antecedent_core::MediationQuery::binary(
        v(population.index("t")),
        v(population.index("y")),
        vec![v(population.index("m"))],
        antecedent_core::MediationContrast::NaturalDirect,
    );
    let builder = Study::tabular(data)
        .query(antecedent_core::CausalQuery::Mediation(query))
        .inference(mode.inference())
        .refute(suite);
    structure.attach(builder, via).build().unwrap()
}

/// Measure the natural direct effect's reported interval on `class` under `mode`.
///
/// The interval is the requested contrast's own: a Frequentist execution reports
/// `estimate ± z·se` from the bootstrap SE of the functional plug-in, a row bootstrap
/// (`bootstrap_se`), except on a `TemporalAdmg`, whose contemporaneous projection resamples
/// circular blocks of consecutive rows and records the row-block family
/// (`circular_block_se`, the same normal interval from that SE); a Bayesian execution reports
/// the quantiles of its Bayesian-bootstrap draws (`posterior_quantile`). A one-atom graph
/// posterior adopts its atom's uncertainty and posterior unchanged.
fn med_measure(
    test: &'static str,
    family: u64,
    class: MedClass,
    mode: Mode,
    variants: &'static [Variant],
) {
    let case = &med_fixture()["covariate_latent"];
    let truth = case["truth"]["natural_direct_effect"].as_f64().unwrap();
    assert_eq!(truth, MED_NATURAL_DIRECT, "the enumerated natural direct effect");
    let population = Population::from_contingency(case);
    assert_eq!(population.size(), 512, "the enumerated table");
    let structure = med_structure(&population, class, variants[0].0 == Posterior);
    let interval = match (mode, class) {
        (Mode::Frequentist, MedClass::TemporalAdmg) => "circular_block_se",
        _ => mode.interval(),
    };
    measure_coordinate(
        &Coordinate { test, dgp: MED_DGP, family, estimator: FUNCTIONAL, interval, variants },
        &Target::scalar(truth),
        |seed, variant| med_study((&population, MED_BASE_N), &structure, mode, seed, variant),
        |result| vec![effect_pair(mode, result)],
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn mediation_admg_given_frequentist_coverage() {
    med_measure(
        "mediation_admg_given_frequentist_coverage",
        0xFE20_0001,
        MedClass::Admg,
        Mode::Frequentist,
        MED_GIVEN,
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn mediation_admg_given_bayesian_coverage() {
    med_measure(
        "mediation_admg_given_bayesian_coverage",
        0xFE20_0002,
        MedClass::Admg,
        Mode::Bayesian,
        MED_GIVEN,
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn mediation_temporal_admg_given_frequentist_coverage() {
    med_measure(
        "mediation_temporal_admg_given_frequentist_coverage",
        0xFE20_0003,
        MedClass::TemporalAdmg,
        Mode::Frequentist,
        MED_GIVEN,
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn mediation_temporal_admg_given_bayesian_coverage() {
    med_measure(
        "mediation_temporal_admg_given_bayesian_coverage",
        0xFE20_0004,
        MedClass::TemporalAdmg,
        Mode::Bayesian,
        MED_GIVEN,
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn mediation_cpdag_given_frequentist_coverage() {
    sharing_base(|| {
        {
            med_measure(
                "mediation_cpdag_given_frequentist_coverage",
                0xFE20_0005,
                MedClass::Cpdag,
                Mode::Frequentist,
                MED_GIVEN,
            );
        }
        {
            med_measure(
                "mediation_cpdag_given_frequentist_coverage",
                0xFE20_0005,
                MedClass::Admg,
                Mode::Frequentist,
                MED_POSTERIOR,
            );
        }
        {
            med_measure(
                "mediation_cpdag_given_frequentist_coverage",
                0xFE20_0005,
                MedClass::Cpdag,
                Mode::Frequentist,
                MED_POSTERIOR,
            );
        }
    });
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn mediation_cpdag_given_bayesian_coverage() {
    sharing_base(|| {
        {
            med_measure(
                "mediation_cpdag_given_bayesian_coverage",
                0xFE20_0006,
                MedClass::Cpdag,
                Mode::Bayesian,
                MED_GIVEN,
            );
        }
        {
            med_measure(
                "mediation_cpdag_given_bayesian_coverage",
                0xFE20_0006,
                MedClass::Admg,
                Mode::Bayesian,
                MED_POSTERIOR,
            );
        }
        {
            med_measure(
                "mediation_cpdag_given_bayesian_coverage",
                0xFE20_0006,
                MedClass::Cpdag,
                Mode::Bayesian,
                MED_POSTERIOR,
            );
        }
    });
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn mediation_temporal_admg_graph_posterior_frequentist_coverage() {
    med_measure(
        "mediation_temporal_admg_graph_posterior_frequentist_coverage",
        0xFE20_0009,
        MedClass::TemporalAdmg,
        Mode::Frequentist,
        MED_POSTERIOR,
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn mediation_temporal_admg_graph_posterior_bayesian_coverage() {
    med_measure(
        "mediation_temporal_admg_graph_posterior_bayesian_coverage",
        0xFE20_000A,
        MedClass::TemporalAdmg,
        Mode::Bayesian,
        MED_POSTERIOR,
    );
}

/// Base sample size of the `markov_parents` PAG design: 256 rows, the rarest cell holds one
/// (`p = 1/256`), the next rarest two. At grid point 0 (`n = 4000`) some cell is missing with
/// probability at most `Σ_cells (1 - p)^4000 ≈ 1.7e-7`.
const MED_PAG_BASE_N: usize = 8_000;

/// The natural direct effect of the `markov_parents` law (`c -> t`, `w -> m`, `t -> m`,
/// `t -> y`, `m -> y`, a DAG over the observed columns `c, w, t, m, y`) on its PAG, every
/// edge directed. The edges out of the roots `c` and `w` are invisible, so the PAG's MAG
/// completion stands for ADMGs that add `c <-> t` or `w <-> m`; their functionals differ as
/// expressions (`execute_pag_path_specific`) and the execution publishes the class identified
/// set. Neither added edge touches `m -> y` or the treatment's effect on `m` and `y`, so every
/// candidate's functional takes the enumerated natural direct effect `5/32` on the table.
fn med_pag_identified_set(test: &'static str, family: u64, mode: Mode) {
    let case = &med_fixture()["markov_parents"];
    let truth = case["truth"]["natural_direct_effect"].as_f64().unwrap();
    assert_eq!(truth, 0.156_25, "the enumerated natural direct effect");
    let population = Population::from_contingency(case);
    assert_eq!(population.size(), 256, "the enumerated table");
    let nodes = u32::try_from(population.names.len()).unwrap();
    let mut pag = Pag::with_variables(nodes);
    for edge in case["graph"]["directed_edges"].as_array().unwrap() {
        let node = |at: usize| d(population.index(edge[at].as_str().unwrap()));
        pag.insert_directed(node(0), node(1)).unwrap();
    }
    let structure = Structure::Pag(pag);
    measure_identified_set(
        &SetCoordinate {
            test,
            dgp: "conformance/estimate/mediation_mixed_graphs::markov_parents",
            family,
            mode,
            variants: MED_GIVEN,
            target: "the population identified set, the point at the natural direct effect \
                     every candidate functional takes on the table",
        },
        truth,
        |seed, variant| med_study((&population, MED_PAG_BASE_N), &structure, mode, seed, variant),
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn mediation_pag_given_frequentist_coverage() {
    med_pag_identified_set(
        "mediation_pag_given_frequentist_coverage",
        0xFE20_000D,
        Mode::Frequentist,
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn mediation_pag_given_bayesian_coverage() {
    med_pag_identified_set("mediation_pag_given_bayesian_coverage", 0xFE20_000E, Mode::Bayesian);
}

// ---- cross-world effects: the counterfactual and the natural direct effect
//
// A counterfactual (the effect on the treated, `E[Y_{x=1} | X = 0] - E[Y | X = 0]`) on a
// latent-confounded class (ADMG, contemporaneous temporal ADMG, PAG) is identified by ID* and
// evaluated on the empirical cell table; a natural direct effect on an ADMG, a temporal ADMG,
// a CPDAG, a PAG or a DAG posterior atom is the path-specific functional along the edges
// that avoid the mediator, evaluated by the finite-discrete plug-in. Both routes publish a
// row-resampled interval: a Frequentist execution the normal interval from the 199-replicate
// row bootstrap SE of the identified functional, a Bayesian one the equal-tailed quantiles of
// the Dirichlet row-weight (Bayesian-bootstrap) posterior of the functional. A Frequentist
// counterfactual resamples only when a replicate count is requested
// (`omitted_bootstrap_resamples` leaves an omitted count at zero, and the route then publishes
// the point with no standard error), so a Frequentist execution requests the default count,
// [`StudyBuilder::OMITTED_BOOTSTRAP`].
//
// Every law is a count table of `conformance/estimate/cross_world_class_truth` (the
// counterfactual graph-posterior coordinates read the same tables from
// `conformance/estimate/cross_world_counterfactual_posterior`). The table is the population,
// a replicate draws its rows uniformly from it, and the fixture's value is the identified
// functional evaluated on the table's own cell counts (in exact rational arithmetic by the
// fixture's `reference.py`), so it is the population value of the reported functional.

/// Base sample size of the 8-cell `front_door` table: its smallest cell has probability
/// `48/2000 = 0.024`, so at the grid's first point (`n = 700`) the chance that any of the 8
/// cells is absent is `8 * 0.976^700 < 1e-6`.
const CW_FRONT_DOOR_N: usize = 1400;

/// Base sample size of the 16- and 32-cell tables (`side_path_chain`, `two_mediators`,
/// `mediated_two_parents`): their smallest cells have probability `8/3200 = 0.0025` and
/// `6/2500 = 0.0024`, so at the grid's first point (`n = 7000`) the chance that any cell is
/// absent is at most `32 * 0.9976^7000 < 1e-6`.
const CW_WIDE_N: usize = 14_000;

/// The cross-world class fixture: the count tables and each case's exact functional.
fn cw_class_pin() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/cross_world_class_truth/expected.json"
    ))
    .unwrap()
}

/// The counterfactual graph-posterior fixture: the same `front_door` and `side_path_chain`
/// tables and their effects on the treated.
fn cw_posterior_pin() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/cross_world_counterfactual_posterior/expected.json"
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

/// The exact value of fixture case `case`.
fn cw_truth(pin: &Value, case: &str) -> f64 {
    pin["cases"][case]["expected"].as_f64().unwrap()
}

/// The effect on the treated of setting `treatment` to 1 on `outcome`.
fn cw_counterfactual(treatment: u32, outcome: u32) -> antecedent_core::CausalQuery {
    antecedent_core::CausalQuery::Counterfactual(antecedent_core::CounterfactualQuery::new(
        v(outcome),
        vec![antecedent_core::Intervention::set(v(treatment), antecedent_core::Value::f64(1.0))],
    ))
}

/// The natural direct effect of `treatment` on `outcome` avoiding `mediator`.
fn cw_nested(treatment: u32, mediator: u32, outcome: u32) -> antecedent_core::CausalQuery {
    antecedent_core::CausalQuery::NestedCounterfactual(
        antecedent_core::NestedCounterfactualQuery::new(v(treatment), v(mediator), v(outcome))
            .unwrap(),
    )
}

fn cw_admg(nodes: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::with_variables(nodes);
    for &(from, to) in directed {
        graph.insert_directed(d(from), d(to)).unwrap();
    }
    for &(a, b) in bidirected {
        graph.insert_bidirected(d(a), d(b)).unwrap();
    }
    graph
}

fn cw_pag(nodes: u32, directed: &[(u32, u32)]) -> Pag {
    let mut graph = Pag::with_variables(nodes);
    for &(from, to) in directed {
        graph.insert_directed(d(from), d(to)).unwrap();
    }
    graph
}

fn cw_cpdag(nodes: u32, directed: &[(u32, u32)]) -> Cpdag {
    let mut graph = Cpdag::with_variables(nodes);
    for &(from, to) in directed {
        graph.insert_directed(d(from), d(to)).unwrap();
    }
    graph
}

/// `t -> m -> y` with `t <-> y` over the `front_door` columns `t, m, y`.
fn cw_front_door_admg() -> Admg {
    cw_admg(3, &[(0, 1), (1, 2)], &[(0, 2)])
}

/// `t -> m`, `t -> y`, `m -> y` with no latent edge over the `front_door` columns.
fn cw_mediator_admg() -> Admg {
    cw_admg(3, &[(0, 1), (0, 2), (1, 2)], &[])
}

/// `w -> x`, `w -> z`, `z -> y`, `x -> y` over the `side_path_chain` columns `w, x, z, y`.
fn cw_side_path_pag() -> Pag {
    cw_pag(4, &[(0, 1), (0, 2), (2, 3), (1, 3)])
}

/// `c -> t`, `w -> m`, `t -> m`, `t -> y`, `m -> y` over the `mediated_two_parents` columns
/// `c, w, t, m, y`.
fn cw_mediated_pag() -> Pag {
    cw_pag(5, &[(0, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
}

/// `t -> a`, `t -> b`, `a -> y`, `b -> y` over the `two_mediators` columns `t, a, b, y`.
fn cw_two_mediators_cpdag() -> Cpdag {
    cw_cpdag(4, &[(0, 1), (0, 2), (1, 3), (2, 3)])
}

/// One cross-world law: the population, the query, the structure and the exact truth.
struct CwDesign {
    population: Population,
    query: antecedent_core::CausalQuery,
    structure: Structure,
    truth: f64,
    base_n: usize,
}

/// The effect on the treated of the `front_door` table on the front-door structure:
/// `sum_m P(m | t=1) P(y=1 | m, t=0) - P(y=1 | t=0) = (0.2 * 0.1 + 0.8 * 0.6) - 0.2 = 0.3`
/// (`P(m=1 | t=1) = 480/600`, `P(y=1 | m=0, t=0) = 112/1120`, `P(y=1 | m=1, t=0) = 168/280`,
/// `P(y=1 | t=0) = 280/1400`). `structure` wraps [`cw_front_door_admg`].
fn cw_front_door_counterfactual(pin: &Value, structure: Structure) -> CwDesign {
    CwDesign {
        population: cw_population(pin, "front_door"),
        query: cw_counterfactual(0, 2),
        structure,
        truth: cw_truth(pin, "admg_counterfactual"),
        base_n: CW_FRONT_DOOR_N,
    }
}

/// The effect on the treated of `x` on `y` in the `side_path_chain` table on its PAG (the
/// edge `x -> y` is visible and `z` is the outcome's one other parent, not a descendant of
/// `x`): `sum_z P(z | x=0) P(y=1 | x=1, z) - P(y=1 | x=0) = (0.65 * 0.4 + 0.35 * 0.8) - 0.24
/// = 0.3`.
fn cw_side_path_counterfactual(pin: &Value, structure: Structure) -> CwDesign {
    CwDesign {
        population: cw_population(pin, "side_path_chain"),
        query: cw_counterfactual(1, 3),
        structure,
        truth: cw_truth(pin, "pag_counterfactual"),
        base_n: CW_WIDE_N,
    }
}

/// The natural direct effect of `t` on `y` avoiding `m` in the `front_door` table on the
/// latent-free mediator structure, by the mediation formula:
/// `sum_m P(m | t=0) [P(y=1 | t=1, m) - P(y=1 | t=0, m)] = 0.8 (0.5 - 0.1) + 0.2 (0.9 - 0.6)
/// = 0.38`.
fn cw_front_door_direct(pin: &Value, structure: Structure) -> CwDesign {
    CwDesign {
        population: cw_population(pin, "front_door"),
        query: cw_nested(0, 1, 2),
        structure,
        truth: cw_truth(pin, "admg_natural_direct"),
        base_n: CW_FRONT_DOOR_N,
    }
}

/// The natural direct effect of `t` on `y` avoiding `m` in the `mediated_two_parents` table
/// on its all-visible PAG: `0.8 (0.5 - 0.1) + 0.2 (0.9 - 0.6) = 0.38`.
fn cw_mediated_direct(pin: &Value, structure: Structure) -> CwDesign {
    CwDesign {
        population: cw_population(pin, "mediated_two_parents"),
        query: cw_nested(2, 3, 4),
        structure,
        truth: cw_truth(pin, "pag_natural_direct"),
        base_n: CW_WIDE_N,
    }
}

/// The natural direct effect of `t` on `y` avoiding the mediator `a` in the `two_mediators`
/// table (the effect along `t -> b -> y`), by the edge g-formula:
/// `sum_a P(a | t=0) sum_b [P(b | t=1) - P(b | t=0)] P(y=1 | a, b)
/// = 0.8 * 0.4 (0.25 - 0.05) + 0.2 * 0.4 (0.8 - 0.4) = 0.096`.
fn cw_two_mediators_direct(pin: &Value, structure: Structure) -> CwDesign {
    CwDesign {
        population: cw_population(pin, "two_mediators"),
        query: cw_nested(0, 1, 3),
        structure,
        truth: cw_truth(pin, "cpdag_path_specific"),
        base_n: CW_WIDE_N,
    }
}

/// Measure `design` as the coordinate `test` with the interval `mode` reports: the
/// bootstrap-SE normal interval of a Frequentist execution, the posterior effect column of a
/// Bayesian one (the population counterfactual's `counterfactual_effect` draws, the
/// functional plug-in's effect draws).
fn cw_measure(
    test: &'static str,
    dgp: &'static str,
    family: u64,
    mode: Mode,
    variants: &[Variant],
    design: &CwDesign,
) {
    measure_coordinate(
        &Coordinate {
            test,
            dgp,
            family,
            estimator: FUNCTIONAL,
            interval: mode.interval(),
            variants,
        },
        &Target::scalar(design.truth),
        |seed, variant| cw_study(test, design, mode, seed, variant),
        |result| vec![effect_pair(mode, result)],
    );
}

/// One execution of `design` on `grid_n(design.base_n)` rows drawn with `seed`.
fn cw_study(test: &str, design: &CwDesign, mode: Mode, seed: u64, (via, suite): Variant) -> Study {
    let n = common::calibration::grid_n(design.base_n);
    let builder = Study::tabular(design.population.sample(n, seed))
        .query(design.query.clone())
        .inference(mode.inference())
        .refute(suite);
    let builder = match mode {
        Mode::Frequentist => builder.bootstrap_replicates(StudyBuilder::OMITTED_BOOTSTRAP),
        Mode::Bayesian => builder,
    };
    design
        .structure
        .attach(builder, via)
        .build()
        .unwrap_or_else(|e| panic!("{test} {via:?} {suite:?}: {e}"))
}

/// Every validation mode of a graph posterior.
const CW_POSTERIOR_ALL: [Variant; 3] = [
    (Posterior, RefuteSuite::None),
    (Posterior, RefuteSuite::Cheap),
    (Posterior, RefuteSuite::Full),
];

/// Every validation mode of a given graph, explicit and accepted.
const CW_GIVEN_ALL: [Variant; 6] = [
    (Explicit, RefuteSuite::None),
    (Accepted, RefuteSuite::None),
    (Explicit, RefuteSuite::Cheap),
    (Accepted, RefuteSuite::Cheap),
    (Explicit, RefuteSuite::Full),
    (Accepted, RefuteSuite::Full),
];

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn counterfactual_admg_graph_posterior_frequentist_coverage() {
    sharing_base(|| {
        {
            let pin = cw_posterior_pin();
            cw_measure(
                "counterfactual_admg_graph_posterior_frequentist_coverage",
                "cross_world_front_door_admg_atom",
                0xFE30_0000,
                Mode::Frequentist,
                &CW_POSTERIOR_ALL,
                &cw_front_door_counterfactual(&pin, Structure::Admg(cw_front_door_admg())),
            );
        }
        {
            let pin = cw_posterior_pin();
            cw_measure(
                "counterfactual_admg_graph_posterior_frequentist_coverage",
                "cross_world_front_door_temporal_admg_atom",
                0xFE30_0000,
                Mode::Frequentist,
                &CW_POSTERIOR_ALL,
                &cw_front_door_counterfactual(
                    &pin,
                    Structure::ContemporaneousAdmg(cw_front_door_admg()),
                ),
            );
        }
    });
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn counterfactual_admg_graph_posterior_bayesian_coverage() {
    sharing_base(|| {
        {
            let pin = cw_posterior_pin();
            cw_measure(
                "counterfactual_admg_graph_posterior_bayesian_coverage",
                "cross_world_front_door_admg_atom",
                0xFE30_0001,
                Mode::Bayesian,
                &CW_POSTERIOR_ALL,
                &cw_front_door_counterfactual(&pin, Structure::Admg(cw_front_door_admg())),
            );
        }
        {
            let pin = cw_posterior_pin();
            cw_measure(
                "counterfactual_admg_graph_posterior_bayesian_coverage",
                "cross_world_front_door_temporal_admg_atom",
                0xFE30_0001,
                Mode::Bayesian,
                &CW_POSTERIOR_ALL,
                &cw_front_door_counterfactual(
                    &pin,
                    Structure::ContemporaneousAdmg(cw_front_door_admg()),
                ),
            );
        }
        {
            let pin = cw_class_pin();
            cw_measure(
                "counterfactual_admg_graph_posterior_bayesian_coverage",
                "cross_world_front_door_admg",
                0xFE30_0001,
                Mode::Bayesian,
                &CW_GIVEN_ALL,
                &cw_front_door_counterfactual(&pin, Structure::Admg(cw_front_door_admg())),
            );
        }
    });
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn counterfactual_pag_graph_posterior_frequentist_coverage() {
    let pin = cw_posterior_pin();
    cw_measure(
        "counterfactual_pag_graph_posterior_frequentist_coverage",
        "cross_world_side_path_pag_atom",
        0xFE30_0004,
        Mode::Frequentist,
        &[(Posterior, RefuteSuite::None)],
        &cw_side_path_counterfactual(&pin, Structure::Pag(cw_side_path_pag())),
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn counterfactual_pag_graph_posterior_bayesian_coverage() {
    sharing_base(|| {
        {
            let pin = cw_posterior_pin();
            cw_measure(
                "counterfactual_pag_graph_posterior_bayesian_coverage",
                "cross_world_side_path_pag_atom",
                0xFE30_0005,
                Mode::Bayesian,
                &[(Posterior, RefuteSuite::None)],
                &cw_side_path_counterfactual(&pin, Structure::Pag(cw_side_path_pag())),
            );
        }
        {
            let pin = cw_class_pin();
            cw_measure(
                "counterfactual_pag_graph_posterior_bayesian_coverage",
                "cross_world_side_path_pag",
                0xFE30_0005,
                Mode::Bayesian,
                &[(Explicit, RefuteSuite::None), (Accepted, RefuteSuite::None)],
                &cw_side_path_counterfactual(&pin, Structure::Pag(cw_side_path_pag())),
            );
        }
    });
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn counterfactual_temporal_admg_given_bayesian_coverage() {
    let pin = cw_class_pin();
    cw_measure(
        "counterfactual_temporal_admg_given_bayesian_coverage",
        "cross_world_front_door_temporal_admg",
        0xFE30_0007,
        Mode::Bayesian,
        &CW_GIVEN_ALL,
        &cw_front_door_counterfactual(&pin, Structure::ContemporaneousAdmg(cw_front_door_admg())),
    );
}

/// The natural direct effect on the `mediated_two_parents` PAG: the edges out of the roots
/// `c` and `w` are invisible, so the PAG admits candidates that add `c <-> t` or `w <-> m`, and
/// those identify the effect by different functionals. The execution publishes the class
/// identified set (product-posterior envelope at 0.90), measured against the natural direct
/// effect 0.38, which every candidate's functional takes on the table.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn nested_counterfactual_effect_pag_given_bayesian_coverage() {
    let pin = cw_class_pin();
    let design = cw_mediated_direct(&pin, Structure::Pag(cw_mediated_pag()));
    let test = "nested_counterfactual_effect_pag_given_bayesian_coverage";
    measure_identified_set(
        &SetCoordinate {
            test,
            dgp: "cross_world_mediated_two_parents_pag",
            family: 0xFE30_000B,
            mode: Mode::Bayesian,
            variants: &CW_GIVEN_ALL,
            target: "the population identified set, the point at the natural direct effect \
                     every candidate functional takes on the table",
        },
        design.truth,
        |seed, variant| cw_study(test, &design, Mode::Bayesian, seed, variant),
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn nested_counterfactual_effect_dag_graph_posterior_bayesian_coverage() {
    sharing_base(|| {
        {
            let pin = cw_class_pin();
            cw_measure(
                "nested_counterfactual_effect_dag_graph_posterior_bayesian_coverage",
                "cross_world_front_door_mediator_dag_atom",
                0xFE30_000D,
                Mode::Bayesian,
                &[(Posterior, RefuteSuite::None), (Posterior, RefuteSuite::Cheap)],
                &cw_front_door_direct(
                    &pin,
                    Structure::Dag { nodes: 3, edges: vec![(0, 1), (0, 2), (1, 2)] },
                ),
            );
        }
        {
            let pin = cw_class_pin();
            cw_measure(
                "nested_counterfactual_effect_dag_graph_posterior_bayesian_coverage",
                "cross_world_front_door_mediator_admg",
                0xFE30_000D,
                Mode::Bayesian,
                &CW_GIVEN_ALL,
                &cw_front_door_direct(&pin, Structure::Admg(cw_mediator_admg())),
            );
        }
        {
            let pin = cw_class_pin();
            cw_measure(
                "nested_counterfactual_effect_dag_graph_posterior_bayesian_coverage",
                "cross_world_front_door_mediator_admg_atom",
                0xFE30_000D,
                Mode::Bayesian,
                &CW_POSTERIOR_ALL,
                &cw_front_door_direct(&pin, Structure::Admg(cw_mediator_admg())),
            );
        }
    });
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn nested_counterfactual_effect_temporal_admg_graph_posterior_bayesian_coverage() {
    sharing_base(|| {
        {
            let pin = cw_class_pin();
            cw_measure(
                "nested_counterfactual_effect_temporal_admg_graph_posterior_bayesian_coverage",
                "cross_world_front_door_mediator_temporal_admg_atom",
                0xFE30_000F,
                Mode::Bayesian,
                &CW_POSTERIOR_ALL,
                &cw_front_door_direct(&pin, Structure::ContemporaneousAdmg(cw_mediator_admg())),
            );
        }
        {
            let pin = cw_class_pin();
            cw_measure(
                "nested_counterfactual_effect_temporal_admg_graph_posterior_bayesian_coverage",
                "cross_world_front_door_mediator_temporal_admg",
                0xFE30_000F,
                Mode::Bayesian,
                &CW_GIVEN_ALL,
                &cw_front_door_direct(&pin, Structure::ContemporaneousAdmg(cw_mediator_admg())),
            );
        }
    });
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn nested_counterfactual_effect_cpdag_graph_posterior_bayesian_coverage() {
    sharing_base(|| {
        {
            let pin = cw_class_pin();
            cw_measure(
                "nested_counterfactual_effect_cpdag_graph_posterior_bayesian_coverage",
                "cross_world_two_mediators_cpdag_atom",
                0xFE30_0010,
                Mode::Bayesian,
                &CW_POSTERIOR_ALL,
                &cw_two_mediators_direct(&pin, Structure::Cpdag(cw_two_mediators_cpdag())),
            );
        }
        {
            let pin = cw_class_pin();
            cw_measure(
                "nested_counterfactual_effect_cpdag_graph_posterior_bayesian_coverage",
                "cross_world_two_mediators_cpdag",
                0xFE30_0010,
                Mode::Bayesian,
                &CW_GIVEN_ALL,
                &cw_two_mediators_direct(&pin, Structure::Cpdag(cw_two_mediators_cpdag())),
            );
        }
    });
}

// ---------------------------------------------------------------- effects
//
// The average effect and the modifier-averaged conditional effect: the front-door
// average effect on an ADMG and on its contemporaneous `TemporalAdmg`, the
// modifier-averaged contrast on a latent-confounded ADMG and its `TemporalAdmg`, and the
// Bayesian average effect on a DAG under both estimators a DAG offers it (`bayesian.gcomp`
// by default, `functional.effect` when the general identifier is selected).
//
// Every execution reports one scalar: the effect estimate (for a conditional effect the
// contrast averaged over the observed modifier law, `Σ_z P(z) τ(z)`; no per-stratum
// interval is published), with the primary interval `primary_interval_binding` selects
// for it. A `TemporalAdmg` functional reads the rows as the consecutive times of one
// series (`Study::functional_row_resampling`): its Frequentist SE is the circular
// row-block bootstrap SE (`FunctionalEffect::estimate` marks the estimate with the
// `RowBlock` family, so the interval is `circular_block_se`), and its Bayesian bootstrap
// shares one Dirichlet weight per block (still `posterior_quantile`). A graph posterior
// of one atom adopts that atom's uncertainty and posterior whole (`atom_mix`,
// `atom_route_posterior`), so it reports the interval its atom reports.

/// Base sample size of the front-door population. Its rarest cell (`t = 1, m = 1,
/// y = 0`) has mass 24/1000, so at the smallest grid point (`n = 700`) some cell is
/// missing with probability at most `8 · 0.976^700 ≈ 3e-7`.
const AE_FRONTDOOR_N: usize = 1400;

/// Base sample size of the modifier-strata population. Its rarest cell has mass 8/180,
/// so at the smallest grid point (`n = 500`) some cell is missing with probability at
/// most `8 · (1 − 8/180)^500 ≈ 1e-9`.
const AE_STRATA_N: usize = 1000;

/// Base sample size of the linear-Gaussian DAG law (as its default-validation
/// coverage test measures it).
const AE_LINEAR_N: usize = 500;

/// `conformance/estimate/admg_frontdoor_functional`: the enumerated binary front-door
/// table (columns `t, m, y`, 1000 rows).
fn ae_frontdoor_fixture() -> &'static Value {
    static FIXTURE: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    FIXTURE.get_or_init(|| {
        serde_json::from_str(include_str!(
            "../../../conformance/estimate/admg_frontdoor_functional/expected.json"
        ))
        .unwrap()
    })
}

/// `conformance/estimate/admg_conditional_strata`: the enumerated `(z, x, y)` table
/// (180 rows) whose two modifier strata carry different effects.
fn ae_strata_fixture() -> &'static Value {
    static FIXTURE: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    FIXTURE.get_or_init(|| {
        serde_json::from_str(include_str!(
            "../../../conformance/estimate/admg_conditional_strata/expected.json"
        ))
        .unwrap()
    })
}

fn ae_frontdoor_population() -> &'static Population {
    static POPULATION: std::sync::OnceLock<Population> = std::sync::OnceLock::new();
    POPULATION.get_or_init(|| Population::from_contingency(ae_frontdoor_fixture()))
}

fn ae_strata_population() -> &'static Population {
    static POPULATION: std::sync::OnceLock<Population> = std::sync::OnceLock::new();
    POPULATION.get_or_init(|| {
        let fixture = ae_strata_fixture();
        Population::from_table(&fixture["columns"], &fixture["table"])
    })
}

/// `n` rows of the front-door population (columns `t`(0), `m`(1), `y`(2)).
fn ae_frontdoor_data(n: usize, seed: u64) -> TabularData {
    ae_frontdoor_population().sample(n, seed)
}

/// `n` rows of the modifier-strata population (columns `z`(0), `x`(1), `y`(2)).
fn ae_strata_data(n: usize, seed: u64) -> TabularData {
    ae_strata_population().sample(n, seed)
}

/// Population front-door average effect of the enumerated table. `P(T = 1) = 0.3`,
/// `P(M = 1 | T = 0) = 0.2`, `P(M = 1 | T = 1) = 0.8`, `P(Y = 1 | M, T)` is `0.1, 0.5` at
/// `M = 0` and `0.6, 0.9` at `M = 1` (for `T = 0, 1`). With
/// `g(m) = Σ_t' P(t') P(Y = 1 | t', m)`: `g(0) = 0.7·0.1 + 0.3·0.5 = 0.22`,
/// `g(1) = 0.7·0.6 + 0.3·0.9 = 0.69`, so the effect is
/// `[0.2 g(0) + 0.8 g(1)] − [0.8 g(0) + 0.2 g(1)] = 0.6 · 0.47 = 0.282` exactly.
fn ae_frontdoor_truth() -> f64 {
    ae_frontdoor_fixture()["frequentist"]["expected_ate"].as_f64().unwrap()
}

/// Population modifier-averaged effect of the strata table:
/// `Σ_z P(z) (E[y | x = 1, z] − E[y | x = 0, z])` with `P(z = 0) = 80/180`, stratum effect
/// `25/40 − 10/40 = 0.375`, and `P(z = 1) = 100/180`, stratum effect `52/60 − 28/40 = 1/6`:
/// `(4/9)(3/8) + (5/9)(1/6) = 7/27`. On the complete DAG `z → x → y`, `z → y` it is also
/// the population average effect of `x` on `y` (the back-door formula over `z`).
fn ae_strata_truth() -> f64 {
    ae_strata_fixture()["expected_ate"].as_f64().unwrap()
}

/// A three-node ADMG with the given directed and bidirected edges.
fn ae_admg(directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::with_variables(3);
    for &(from, to) in directed {
        graph.insert_directed(d(from), d(to)).unwrap();
    }
    for &(a, b) in bidirected {
        graph.insert_bidirected(d(a), d(b)).unwrap();
    }
    graph
}

/// The front-door ADMG `t → m → y`, `t ↔ y`.
fn ae_frontdoor_admg() -> Admg {
    ae_admg(&[(0, 1), (1, 2)], &[(0, 2)])
}

/// `z → x`, `z → y`, `x → y`, `z ↔ x`: a latent common cause of the modifier and the
/// treatment; conditioning on `z` closes every back-door path of `x → y`.
fn ae_bow_admg() -> Admg {
    ae_admg(&[(0, 1), (0, 2), (1, 2)], &[(0, 1)])
}

/// `z → x → y`, `z ↔ y`: latent-confounded without a bow, so a posterior adjacency atom
/// encodes it; conditioning on `z` closes `x ← z ↔ y`, and the identified contrast is the
/// same stratified difference.
fn ae_posterior_admg() -> Admg {
    ae_admg(&[(0, 1), (1, 2)], &[(0, 2)])
}

/// One law of this family: how a replicate's rows are drawn, the structure and query
/// they are analysed under, and the estimator and truth of the reported effect.
struct AeDesign {
    dgp: &'static str,
    base_n: usize,
    data: fn(usize, u64) -> TabularData,
    structure: Structure,
    query: antecedent_core::CausalQuery,
    /// Select the general identifier (routes a DAG average effect to `functional.effect`).
    general_id: bool,
    /// The structure is a `TemporalAdmg`: the Frequentist SE is a row-block bootstrap SE.
    temporal: bool,
    estimator: &'static str,
    truth: f64,
}

fn ae_frontdoor_query() -> antecedent_core::CausalQuery {
    antecedent_core::CausalQuery::AverageEffect(antecedent_core::AverageEffectQuery::with_levels(
        v(0),
        v(2),
        0.0,
        1.0,
    ))
}

fn ae_conditional_query() -> antecedent_core::CausalQuery {
    antecedent_core::CausalQuery::ConditionalEffect(
        antecedent_core::ConditionalEffectQuery::try_new(
            antecedent_core::AverageEffectQuery::binary_ate(v(1), v(2))
                .with_effect_modifiers(vec![v(0)]),
        )
        .unwrap(),
    )
}

/// The front-door average effect on the ADMG, or on its contemporaneous `TemporalAdmg`.
fn ae_frontdoor_design(temporal: bool) -> AeDesign {
    let graph = ae_frontdoor_admg();
    AeDesign {
        dgp: "admg_frontdoor_functional_population",
        base_n: AE_FRONTDOOR_N,
        data: ae_frontdoor_data,
        structure: if temporal {
            Structure::ContemporaneousAdmg(graph)
        } else {
            Structure::Admg(graph)
        },
        query: ae_frontdoor_query(),
        general_id: false,
        temporal,
        estimator: FUNCTIONAL,
        truth: ae_frontdoor_truth(),
    }
}

/// The modifier-averaged conditional effect: a given graph is the bow ADMG, a posterior
/// atom the bow-free ADMG; `temporal` lifts either to its contemporaneous `TemporalAdmg`.
fn ae_conditional_design(temporal: bool, posterior: bool) -> AeDesign {
    let graph = if posterior { ae_posterior_admg() } else { ae_bow_admg() };
    AeDesign {
        dgp: "admg_conditional_strata_population",
        base_n: AE_STRATA_N,
        data: ae_strata_data,
        structure: if temporal {
            Structure::ContemporaneousAdmg(graph)
        } else {
            Structure::Admg(graph)
        },
        query: ae_conditional_query(),
        general_id: false,
        temporal,
        estimator: FUNCTIONAL,
        truth: ae_strata_truth(),
    }
}

/// The Bayesian DAG average effect under its default estimator `bayesian.gcomp`: the
/// linear-Gaussian law of `common::static_dgp::linear_ate_data` (columns `t`(0), `y`(1),
/// `z`(2); `z ~ N(0,1)`, `t ~ Bern(σ(−0.8 + z))`, `y = 2t + z + N(0,1)`), whose average
/// effect is exactly 2 on the DAG `z → t`, `z → y`, `t → y`.
fn ae_dag_gcomp_design() -> AeDesign {
    AeDesign {
        dgp: "crates/antecedent/tests/common/static_dgp.rs::linear_ate_data",
        base_n: AE_LINEAR_N,
        data: common::static_dgp::linear_ate_data,
        structure: Structure::Dag { nodes: 3, edges: vec![(2, 0), (2, 1), (0, 1)] },
        query: antecedent_core::CausalQuery::AverageEffect(
            antecedent_core::AverageEffectQuery::binary_ate(v(0), v(1)),
        ),
        general_id: false,
        temporal: false,
        estimator: "bayesian.gcomp",
        truth: 2.0,
    }
}

/// The Bayesian DAG average effect routed to `functional.effect` by selecting the general
/// identifier: the strata population on the complete DAG `z → x`, `z → y`, `x → y`. Every
/// law is Markov to a complete DAG, so the identified functional
/// `Σ_z P(z) E[y | x, z]` is the population interventional mean and the average effect of
/// `x` on `y` is the stratified difference 7/27.
fn ae_dag_functional_design() -> AeDesign {
    AeDesign {
        dgp: "admg_conditional_strata_population",
        base_n: AE_STRATA_N,
        data: ae_strata_data,
        structure: Structure::Dag { nodes: 3, edges: vec![(0, 1), (0, 2), (1, 2)] },
        query: antecedent_core::CausalQuery::AverageEffect(
            antecedent_core::AverageEffectQuery::with_levels(v(1), v(2), 0.0, 1.0),
        ),
        general_id: true,
        temporal: false,
        estimator: FUNCTIONAL,
        truth: ae_strata_truth(),
    }
}

/// Measure `design` under `mode` on `variants`, scoring the effect's reported interval.
fn ae_measure(
    test: &'static str,
    family: u64,
    design: &AeDesign,
    mode: Mode,
    variants: &[Variant],
) {
    let interval = match (mode, design.temporal) {
        (Mode::Frequentist, true) => "circular_block_se",
        _ => mode.interval(),
    };
    measure_coordinate(
        &Coordinate {
            test,
            dgp: design.dgp,
            family,
            estimator: design.estimator,
            interval,
            variants,
        },
        &Target::scalar(design.truth),
        |seed, (via, suite)| {
            let data = (design.data)(common::calibration::grid_n(design.base_n), seed);
            let builder = design
                .structure
                .attach(Study::tabular(data), via)
                .query(design.query.clone())
                .inference(mode.inference())
                .refute(suite);
            let builder = if design.general_id {
                builder.identifier(antecedent::IdentifierId::GeneralId)
            } else {
                builder
            };
            builder.build().unwrap_or_else(|e| panic!("{test} {via:?} {suite:?}: {e}"))
        },
        |result| match (mode, design.estimator) {
            (Mode::Bayesian, "bayesian.gcomp") => vec![gcomp_rank_pair(result)],
            _ => vec![effect_pair(mode, result)],
        },
    );
}

/// `[reported, gate]` of a `bayesian.gcomp` Laplace posterior.
///
/// Below 1,000 draws (`LAPLACE_RANK_INTERVAL_BELOW_DRAWS`; this file draws [`DRAWS`]) the
/// estimator replaces the effect column's type-7 `q025`/`q975` summaries with the
/// exchangeable-rank (Hyndman-Fan type 6) equal-tailed interval of the same draws, whose
/// expected posterior mass is the level at every draw count (`summarize_gcomp_draws`). The
/// published interval is checked to be that rule at 0.95 on the retained draws bit for bit,
/// and the gate interval is the same rule at 0.90 on the same draws.
fn gcomp_rank_pair(result: &StudyResult) -> IntervalPair {
    let posterior = result.posterior.as_ref().expect("a posterior");
    let column = posterior.effect_column().expect("a posterior effect column");
    let reported = (posterior.summaries.q025[column], posterior.summaries.q975[column]);
    let mut sorted = posterior.draws.column(column).expect("the effect draws").to_vec();
    assert!(
        sorted.len() >= 2 && sorted.iter().all(|value| value.is_finite()),
        "finite effect draws"
    );
    sorted.sort_by(f64::total_cmp);
    let at = |level: f64| {
        antecedent_stats::equal_tail_interval_sorted(
            &sorted,
            level,
            antecedent_stats::QuantileRule::ExchangeableRank,
        )
    };
    let rebuilt = at(REPORTED_LEVEL);
    assert_eq!(
        (rebuilt.0.to_bits(), rebuilt.1.to_bits()),
        (reported.0.to_bits(), reported.1.to_bits()),
        "the published interval [{}, {}] is not the exchangeable-rank interval of its draws \
         [{}, {}]",
        reported.0,
        reported.1,
        rebuilt.0,
        rebuilt.1
    );
    [Some(reported), Some(at(GATE_LEVEL))]
}

/// The ADMG front-door average effect, Frequentist bootstrap-SE interval, given graph.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn average_effect_admg_given_frequentist_coverage() {
    sharing_base(|| {
        {
            ae_measure(
                "average_effect_admg_given_frequentist_coverage",
                0xFE40_0000,
                &ae_frontdoor_design(false),
                Mode::Frequentist,
                &[
                    (Accepted, RefuteSuite::None),
                    (Explicit, RefuteSuite::Cheap),
                    (Accepted, RefuteSuite::Cheap),
                    (Explicit, RefuteSuite::Full),
                    (Accepted, RefuteSuite::Full),
                ],
            );
        }
        // The ADMG front-door average effect, Frequentist, one-atom ADMG graph posterior.
        {
            ae_measure(
                "average_effect_admg_given_frequentist_coverage",
                0xFE40_0000,
                &ae_frontdoor_design(false),
                Mode::Frequentist,
                &[(Posterior, RefuteSuite::Cheap), (Posterior, RefuteSuite::Full)],
            );
        }
    });
}

/// The ADMG front-door average effect, Bayesian, one-atom ADMG graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn average_effect_admg_graph_posterior_bayesian_coverage() {
    sharing_base(|| {
        {
            ae_measure(
                "average_effect_admg_graph_posterior_bayesian_coverage",
                0xFE40_0003,
                &ae_frontdoor_design(false),
                Mode::Bayesian,
                &[(Posterior, RefuteSuite::Cheap), (Posterior, RefuteSuite::Full)],
            );
        }
        // The ADMG front-door average effect, Bayesian-bootstrap posterior, given graph.
        {
            ae_measure(
                "average_effect_admg_graph_posterior_bayesian_coverage",
                0xFE40_0003,
                &ae_frontdoor_design(false),
                Mode::Bayesian,
                &[
                    (Accepted, RefuteSuite::None),
                    (Explicit, RefuteSuite::Cheap),
                    (Accepted, RefuteSuite::Cheap),
                    (Explicit, RefuteSuite::Full),
                    (Accepted, RefuteSuite::Full),
                ],
            );
        }
    });
}

/// The Bayesian DAG average effect under `bayesian.gcomp` (the default estimator).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn average_effect_dag_given_bayesian_bayesian_gcomp_coverage() {
    ae_measure(
        "average_effect_dag_given_bayesian_bayesian_gcomp_coverage",
        0xFE40_0004,
        &ae_dag_gcomp_design(),
        Mode::Bayesian,
        &[
            (Explicit, RefuteSuite::Cheap),
            (Accepted, RefuteSuite::Cheap),
            (Explicit, RefuteSuite::Full),
            (Accepted, RefuteSuite::Full),
        ],
    );
}

/// The Bayesian DAG average effect under `functional.effect` (general identifier).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn average_effect_dag_given_bayesian_functional_effect_coverage() {
    ae_measure(
        "average_effect_dag_given_bayesian_functional_effect_coverage",
        0xFE40_0005,
        &ae_dag_functional_design(),
        Mode::Bayesian,
        &[
            (Explicit, RefuteSuite::Cheap),
            (Accepted, RefuteSuite::Cheap),
            (Explicit, RefuteSuite::Full),
            (Accepted, RefuteSuite::Full),
        ],
    );
}

/// The front-door average effect on the contemporaneous `TemporalAdmg`, Frequentist
/// circular row-block bootstrap-SE interval, given graph.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn average_effect_temporal_admg_given_frequentist_coverage() {
    sharing_base(|| {
        {
            ae_measure(
                "average_effect_temporal_admg_given_frequentist_coverage",
                0xFE40_0006,
                &ae_frontdoor_design(true),
                Mode::Frequentist,
                &[
                    (Explicit, RefuteSuite::None),
                    (Accepted, RefuteSuite::None),
                    (Explicit, RefuteSuite::Cheap),
                    (Accepted, RefuteSuite::Cheap),
                    (Explicit, RefuteSuite::Full),
                    (Accepted, RefuteSuite::Full),
                ],
            );
        }
        // The front-door average effect, Frequentist, one-atom `TemporalAdmg` graph posterior.
        {
            ae_measure(
                "average_effect_temporal_admg_given_frequentist_coverage",
                0xFE40_0006,
                &ae_frontdoor_design(true),
                Mode::Frequentist,
                &[
                    (Posterior, RefuteSuite::None),
                    (Posterior, RefuteSuite::Cheap),
                    (Posterior, RefuteSuite::Full),
                ],
            );
        }
    });
}

/// The front-door average effect on the contemporaneous `TemporalAdmg`, block Bayesian
/// bootstrap posterior, given graph.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn average_effect_temporal_admg_given_bayesian_coverage() {
    sharing_base(|| {
        {
            ae_measure(
                "average_effect_temporal_admg_given_bayesian_coverage",
                0xFE40_0008,
                &ae_frontdoor_design(true),
                Mode::Bayesian,
                &[
                    (Explicit, RefuteSuite::None),
                    (Accepted, RefuteSuite::None),
                    (Explicit, RefuteSuite::Cheap),
                    (Accepted, RefuteSuite::Cheap),
                    (Explicit, RefuteSuite::Full),
                    (Accepted, RefuteSuite::Full),
                ],
            );
        }
        // The front-door average effect, Bayesian, one-atom `TemporalAdmg` graph posterior.
        {
            ae_measure(
                "average_effect_temporal_admg_given_bayesian_coverage",
                0xFE40_0008,
                &ae_frontdoor_design(true),
                Mode::Bayesian,
                &[
                    (Posterior, RefuteSuite::None),
                    (Posterior, RefuteSuite::Cheap),
                    (Posterior, RefuteSuite::Full),
                ],
            );
        }
    });
}

/// The modifier-averaged conditional effect on the bow ADMG, Frequentist bootstrap-SE
/// interval, given graph.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn conditional_effect_admg_given_frequentist_coverage() {
    ae_measure(
        "conditional_effect_admg_given_frequentist_coverage",
        0xFE40_000A,
        &ae_conditional_design(false, false),
        Mode::Frequentist,
        &[(Explicit, RefuteSuite::None), (Accepted, RefuteSuite::None)],
    );
}

/// The modifier-averaged conditional effect on the bow ADMG, Bayesian, given graph.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn conditional_effect_admg_given_bayesian_coverage() {
    ae_measure(
        "conditional_effect_admg_given_bayesian_coverage",
        0xFE40_000B,
        &ae_conditional_design(false, false),
        Mode::Bayesian,
        &[
            (Explicit, RefuteSuite::None),
            (Accepted, RefuteSuite::None),
            (Explicit, RefuteSuite::Cheap),
            (Accepted, RefuteSuite::Cheap),
            (Explicit, RefuteSuite::Full),
            (Accepted, RefuteSuite::Full),
        ],
    );
}

/// The modifier-averaged conditional effect, Bayesian, one-atom ADMG graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn conditional_effect_admg_graph_posterior_bayesian_coverage() {
    sharing_base(|| {
        {
            ae_measure(
                "conditional_effect_admg_graph_posterior_bayesian_coverage",
                0xFE40_000C,
                &ae_conditional_design(false, true),
                Mode::Bayesian,
                &[
                    (Posterior, RefuteSuite::None),
                    (Posterior, RefuteSuite::Cheap),
                    (Posterior, RefuteSuite::Full),
                ],
            );
        }
        // The modifier-averaged conditional effect, Bayesian, one-atom `TemporalAdmg` graph
        // posterior.
        {
            ae_measure(
                "conditional_effect_admg_graph_posterior_bayesian_coverage",
                0xFE40_000C,
                &ae_conditional_design(true, true),
                Mode::Bayesian,
                &[
                    (Posterior, RefuteSuite::None),
                    (Posterior, RefuteSuite::Cheap),
                    (Posterior, RefuteSuite::Full),
                ],
            );
        }
    });
}

/// The modifier-averaged conditional effect on the contemporaneous `TemporalAdmg` of the
/// bow ADMG, block Bayesian bootstrap posterior, given graph.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn conditional_effect_temporal_admg_given_bayesian_coverage() {
    ae_measure(
        "conditional_effect_temporal_admg_given_bayesian_coverage",
        0xFE40_000D,
        &ae_conditional_design(true, false),
        Mode::Bayesian,
        &[
            (Explicit, RefuteSuite::None),
            (Accepted, RefuteSuite::None),
            (Explicit, RefuteSuite::Cheap),
            (Accepted, RefuteSuite::Cheap),
            (Explicit, RefuteSuite::Full),
            (Accepted, RefuteSuite::Full),
        ],
    );
}

// ---- response: InterventionResponse and ResponseCurve on Admg and TemporalAdmg
//
// The intervention level `E[Y | do(T = t)]` and the mean curve over a grid of levels, on a
// given ADMG (explicit or accepted), its contemporaneous `TemporalAdmg` lift over the same
// table, and a one-atom graph posterior of either kind.
//
// A given mixed graph routes by its edges (`compile_logical_mixed_admg_response`): a graph
// with a bidirected edge is identified by the general ID algorithm and evaluated by the
// `functional.effect` plug-in; a graph with none is the DAG of its directed edges and takes
// the static response estimator (`response.intervention_gcomp` for a Frequentist level,
// `response.kennedy_dr` for a Frequentist curve, `response.bayesian` under Bayesian
// inference). Each of the two routes is measured on its own law:
//
// * [`rs_frontdoor_mean`]: the front-door table of
//   `conformance/estimate/admg_frontdoor_functional` (`T -> M -> Y`, `T <-> Y`), sampled as an
//   exact finite population. Its front-door levels are `E[Y | do(T = 0)] = 0.314` and
//   `E[Y | do(T = 1)] = 0.596`.
// * [`rs_adjustable_data`]: `z -> t -> y` with no bidirected edge, linear in `t`, with
//   `E[Y | do(T = a)] = 1 + 2a`.
//
// Intervals (`StudyResult::primary_interval_binding`):
//
// * a Frequentist `functional.effect` level or curve publishes `value ± z·se` from the
//   plug-in bootstrap SE of every grid level (support diagnostic `response.bootstrap_se`,
//   reported `bootstrap_se`). A Response query resamples only when a replicate count is
//   requested (`omitted_bootstrap_resamples`), so these executions request the default
//   count, [`StudyBuilder::OMITTED_BOOTSTRAP`]. The 0.90 interval is the same construction
//   formed from the published value and the SE the published half-width states;
// * a Bayesian `functional.effect` level or curve publishes the `q025`/`q975` summaries of
//   each level's Bayesian-bootstrap draws (`posterior_quantile`). A level's draws are on the
//   result; a curve that keeps only its first level's draws (a graph posterior, a validated
//   explicit curve) is read through single-level executions of the same structure route, each
//   required to reproduce the published interval of its grid point bit for bit;
// * `response.intervention_gcomp` publishes an analytic `± z·se` scalar, `response.bayesian`
//   retains the draws behind its credible interval or band, and the `response.kennedy_dr`
//   analytic band is refitted on the same data and seed at the 0.90 level, which changes only
//   the interval quantile.

/// Base sample size of the front-door designs. The rarest table cell has mass
/// `24 / 1000`, so at the smallest grid point (700 rows) every one of the eight cells is
/// present with probability above `1 - 8·0.976^700 ≈ 1 - 3.3e-7`.
const RS_FRONTDOOR_N: usize = 1400;

/// Base sample size of the adjustment-identified designs.
const RS_ADJUSTABLE_N: usize = 500;

/// The level of every `InterventionResponse` design: `do(T = 1)`.
const RS_LEVEL: f64 = 1.0;

/// The front-door curve's grid: the two treatment levels of the binary table.
const RS_FRONTDOOR_GRID: [f64; 2] = [0.0, 1.0];

/// The adjustment-identified curve's grid.
const RS_ADJUSTABLE_GRID: [f64; 3] = [-1.0, 0.0, 1.0];

/// Store tag of a credible interval read from a single-level execution.
const RS_CREDIBLE_TAG: u64 = 0xC4ED_1B1E;

/// Store tag of a band refitted at the gate level.
const RS_REFIT_TAG: u64 = 0x4EF1_7000;

/// The law a design samples, which also fixes the estimator its graph routes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RsLaw {
    /// The front-door table on `T -> M -> Y`, `T <-> Y`: general ID, `functional.effect`.
    FrontDoor,
    /// [`rs_adjustable_data`] on `z -> t -> y`: adjustment, the static response estimator.
    Adjustable,
}

/// The graph class a design is given as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RsClass {
    Admg,
    /// The contemporaneous `TemporalAdmg` lift of the same ADMG over the same table.
    TemporalAdmg,
}

/// The response functional of a design.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RsQuery {
    /// `InterventionResponse` at [`RS_LEVEL`].
    Level,
    /// `MeanCurve` over the law's grid.
    Curve,
}

/// Everything one coordinate executes and scores.
struct RsSpec {
    law: RsLaw,
    query: RsQuery,
    dgp: &'static str,
    base_n: usize,
    /// The front-door population; `None` for [`RsLaw::Adjustable`].
    population: Option<Population>,
    structure: Structure,
    treatment: u32,
    outcome: u32,
    /// The grid of a curve, or the single level of an `InterventionResponse`.
    grid: Vec<f64>,
    targets: Vec<Target>,
    estimator: &'static str,
    interval: &'static str,
}

impl RsSpec {
    /// The replicate's rows at this run's grid point.
    fn sample(&self, seed: u64) -> TabularData {
        let n = common::calibration::grid_n(self.base_n);
        match &self.population {
            Some(population) => population.sample(n, seed),
            None => rs_adjustable_data(n, seed),
        }
    }
}

// ------------------------------------------------------------------- the laws

fn rs_frontdoor_pin() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/admg_frontdoor_functional/expected.json"
    ))
    .unwrap()
}

/// The fixture's ADMG over the population's columns.
fn rs_frontdoor_graph(pin: &Value, population: &Population) -> Admg {
    let mut graph = Admg::with_variables(u32::try_from(population.names.len()).unwrap());
    let node = |name: &Value| d(population.index(name.as_str().unwrap()));
    for edge in pin["graph"]["directed_edges"].as_array().unwrap() {
        graph.insert_directed(node(&edge[0]), node(&edge[1])).unwrap();
    }
    for edge in pin["graph"]["bidirected_edges"].as_array().unwrap() {
        graph.insert_bidirected(node(&edge[0]), node(&edge[1])).unwrap();
    }
    graph
}

/// The front-door level `E[Y | do(T = level)] = Σ_m P(m | level) Σ_t' P(t') P(Y = 1 | t', m)`
/// of the enumerated table, from its counts.
///
/// The table has `P(T = 1) = 0.3`, `P(M = 1 | T = 0) = 0.2`, `P(M = 1 | T = 1) = 0.8`, and
/// `P(Y = 1 | T, M)` of `0.1, 0.5` at `M = 0` and `0.6, 0.9` at `M = 1` (for `T = 0, 1`), so
/// `g(0) = 0.7·0.1 + 0.3·0.5 = 0.22`, `g(1) = 0.7·0.6 + 0.3·0.9 = 0.69` and
/// `E[Y | do(T = 0)] = 0.8·0.22 + 0.2·0.69 = 0.314`,
/// `E[Y | do(T = 1)] = 0.2·0.22 + 0.8·0.69 = 0.596`. A replicate samples the table's rows, so
/// these are the population values of the functional the plug-in evaluates.
fn rs_frontdoor_mean(pin: &Value, level: f64) -> f64 {
    let mut count = [[[0.0_f64; 2]; 2]; 2];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let at = |name: &str| cell[name].as_f64().unwrap() as usize;
        count[at("t")][at("m")][at("y")] += cell["count"].as_f64().unwrap();
    }
    let total: f64 = count.iter().flatten().flatten().sum();
    let arm = |t: usize| count[t].iter().flatten().sum::<f64>();
    let stratum = |t: usize, m: usize| count[t][m][0] + count[t][m][1];
    let level = level as usize;
    (0..2)
        .map(|m| {
            let inner: f64 = (0..2).map(|t| arm(t) / total * count[t][m][1] / stratum(t, m)).sum();
            stratum(level, m) / arm(level) * inner
        })
        .sum()
}

/// `[E[Y | do(T = 0)], E[Y | do(T = 1)]]` of the front-door table, checked against both
/// fixtures that state them: the contrast is `admg_frontdoor_functional`'s `expected_ate`
/// 0.282 and the levels are `response_curve_posterior_refuter_states`'s front-door curve.
fn rs_frontdoor_truths(pin: &Value) -> [f64; 2] {
    let truths = RS_FRONTDOOR_GRID.map(|level| rs_frontdoor_mean(pin, level));
    let ate = pin["frequentist"]["expected_ate"].as_f64().unwrap();
    assert!((truths[1] - truths[0] - ate).abs() < 1e-12, "front-door contrast {truths:?}");
    let states: Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/response_curve_posterior_refuter_states/expected.json"
    ))
    .unwrap();
    let closed = &states["closed_form"];
    for (j, truth) in truths.iter().enumerate() {
        assert_eq!(closed["grid"][j].as_f64(), Some(RS_FRONTDOOR_GRID[j]), "fixture grid");
        let stated = closed["frontdoor_curve"][j].as_f64().unwrap();
        assert!((truth - stated).abs() < 1e-12, "front-door level {j}: {truth} vs {stated}");
    }
    truths
}

/// Columns `t`(0), `y`(1), `z`(2): `z ~ N(0,1)`, `t = 0.6 z + 0.8 e`, `y = 1 + 2t + e`.
///
/// `z` causes the treatment only, so the empty set and `{z}` are both valid adjustment sets
/// and the outcome does not depend on `z` given `t`: `E[Y | do(T = a)] = 1 + 2a` under either,
/// and a response that holds the empirical law of `z` fixed targets the same value.
fn rs_adjustable_data(n: usize, seed: u64) -> TabularData {
    let mut g = common::calibration::gaussian(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        t[i] = 0.6 * z[i] + 0.8 * g();
        y[i] = 1.0 + 2.0 * t[i] + g();
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

/// `E[Y | do(T = a)]` of [`rs_adjustable_data`].
fn rs_adjustable_mean(a: f64) -> f64 {
    1.0 + 2.0 * a
}

/// `z -> t`, `t -> y` over [`rs_adjustable_data`]'s columns: no bidirected edge.
fn rs_adjustable_admg() -> Admg {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(d(2), d(0)).unwrap();
    graph.insert_directed(d(0), d(1)).unwrap();
    graph
}

fn rs_spec(law: RsLaw, class: RsClass, query: RsQuery, mode: Mode) -> RsSpec {
    let lift = |graph: Admg| match class {
        RsClass::Admg => Structure::Admg(graph),
        RsClass::TemporalAdmg => Structure::ContemporaneousAdmg(graph),
    };
    let labelled = |grid: &[f64], truth: &dyn Fn(usize, f64) -> f64| -> Vec<Target> {
        grid.iter()
            .enumerate()
            .map(|(j, &a)| Target::labelled(format!("t={a}"), truth(j, a)))
            .collect()
    };
    match law {
        RsLaw::FrontDoor => {
            let pin = rs_frontdoor_pin();
            let population = Population::from_contingency(&pin);
            let graph = rs_frontdoor_graph(&pin, &population);
            let truths = rs_frontdoor_truths(&pin);
            let (grid, targets) = match query {
                RsQuery::Level => {
                    (vec![RS_LEVEL], Target::scalar(rs_frontdoor_mean(&pin, RS_LEVEL)))
                }
                RsQuery::Curve => {
                    (RS_FRONTDOOR_GRID.to_vec(), labelled(&RS_FRONTDOOR_GRID, &|j, _| truths[j]))
                }
            };
            RsSpec {
                law,
                query,
                dgp: "admg_frontdoor_functional",
                base_n: RS_FRONTDOOR_N,
                treatment: population.index("t"),
                outcome: population.index("y"),
                population: Some(population),
                structure: lift(graph),
                grid,
                targets,
                estimator: FUNCTIONAL,
                interval: mode.interval(),
            }
        }
        RsLaw::Adjustable => {
            let (grid, targets) = match query {
                RsQuery::Level => (vec![RS_LEVEL], Target::scalar(rs_adjustable_mean(RS_LEVEL))),
                RsQuery::Curve => (
                    RS_ADJUSTABLE_GRID.to_vec(),
                    labelled(&RS_ADJUSTABLE_GRID, &|_, a| rs_adjustable_mean(a)),
                ),
            };
            let (estimator, interval) = match (query, mode) {
                (_, Mode::Bayesian) => ("response.bayesian", "posterior_quantile"),
                (RsQuery::Level, Mode::Frequentist) => {
                    ("response.intervention_gcomp", "analytic_se")
                }
                (RsQuery::Curve, Mode::Frequentist) => ("response.kennedy_dr", "analytic_se"),
            };
            RsSpec {
                law,
                query,
                dgp: "adjustable_admg_data",
                base_n: RS_ADJUSTABLE_N,
                population: None,
                structure: lift(rs_adjustable_admg()),
                treatment: 0,
                outcome: 1,
                grid,
                targets,
                estimator,
                interval,
            }
        }
    }
}

// -------------------------------------------------------------- the executions

/// One execution of `spec` on `data`: the `query` over `grid` (a single level for
/// [`RsQuery::Level`]), with a response band at `level` where the estimator takes one.
fn rs_study(
    spec: &RsSpec,
    mode: Mode,
    (via, suite): Variant,
    data: TabularData,
    (query, grid): (RsQuery, &[f64]),
    level: f64,
) -> Study {
    let (treatment, outcome) = (v(spec.treatment), v(spec.outcome));
    let functional = match query {
        RsQuery::Level => antecedent_core::ResponseFunctional::InterventionResponse {
            outcome,
            interventions: std::sync::Arc::from([antecedent_core::Intervention::set(
                treatment,
                antecedent_core::Value::f64(grid[0]),
            )]),
        },
        RsQuery::Curve => antecedent_core::ResponseFunctional::MeanCurve {
            outcome,
            treatment: antecedent_core::ContinuousDomain::new(
                treatment,
                antecedent_core::GridSpec::Values(grid.to_vec().into()),
            ),
        },
    };
    let builder = spec
        .structure
        .attach(Study::tabular(data), via)
        .query(antecedent_core::CausalQuery::Response(antecedent_core::ResponseQuery::new(
            functional,
        )))
        .inference(mode.inference())
        .refute(suite);
    let builder = match (spec.law, mode) {
        // A Response query resamples only when a replicate count is requested; the
        // general-ID band is the plug-in bootstrap SE at the default count.
        (RsLaw::FrontDoor, Mode::Frequentist) => {
            builder.bootstrap_replicates(StudyBuilder::OMITTED_BOOTSTRAP)
        }
        (RsLaw::FrontDoor, Mode::Bayesian) => builder.bootstrap_replicates(0),
        (RsLaw::Adjustable, _) => builder.bootstrap_replicates(0).response_options(
            antecedent_estimate::ContinuousResponseOptions {
                confidence_level: level,
                ..antecedent_estimate::ContinuousResponseOptions::default()
            },
        ),
    };
    builder.build().unwrap_or_else(|e| panic!("{} {via:?} {suite:?} {grid:?}: {e}", spec.dgp))
}

fn rs_execute(
    spec: &RsSpec,
    mode: Mode,
    execution: Variant,
    data: &TabularData,
    shape: (RsQuery, &[f64]),
    level: f64,
    seed: u64,
) -> StudyResult {
    rs_study(spec, mode, execution, data.clone(), shape, level)
        .run(&ExecutionContext::for_tests(seed))
        .unwrap_or_else(|e| panic!("{} {execution:?} {:?}: {e}", spec.dgp, shape.1))
}

/// Every reported coordinate of a point-identified response.
fn rs_values(result: &StudyResult) -> Vec<f64> {
    use antecedent_core::{ResponseIdentification, ResponseValue};
    match &result.response().expect("a response result").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => vec![*value],
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
            mean.to_vec()
        }
        other => panic!("expected a point-identified level or curve, got {other:?}"),
    }
}

type RsGates = std::sync::Mutex<std::collections::HashMap<Vec<u64>, Vec<Option<(f64, f64)>>>>;

/// Gate-level intervals of reported intervals that one execution cannot re-form, keyed by
/// what identifies them: the bits of a credible interval a single-level execution reproduced
/// ([`RS_CREDIBLE_TAG`]), or the estimate of an execution refitted at the gate level
/// ([`RS_REFIT_TAG`]). A sibling that reports the base interval bit for bit reads the base
/// execution's entry; one that does not finds none.
fn rs_gates() -> &'static RsGates {
    static STORE: std::sync::OnceLock<RsGates> = std::sync::OnceLock::new();
    STORE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn rs_credible_key(lower: f64, upper: f64) -> Vec<u64> {
    vec![RS_CREDIBLE_TAG, lower.to_bits(), upper.to_bits()]
}

fn rs_refit_key(result: &StudyResult) -> Vec<u64> {
    std::iter::once(RS_REFIT_TAG).chain(rs_values(result).iter().map(|x| x.to_bits())).collect()
}

/// The posterior column behind a credible level: the effect column, or the only column
/// (the rule `credible_scalar_uncertainty` publishes with).
fn rs_credible_column(result: &StudyResult) -> usize {
    let posterior = result.posterior.as_ref().expect("the draws behind the credible interval");
    posterior
        .effect_column()
        .or_else(|| (posterior.draws.schema.n_quantities() == 1).then_some(0))
        .expect("a posterior effect column")
}

/// `[reported, gate]` of a Bayesian `functional.effect` level: the published credible
/// interval, which must be the `q025`/`q975` summaries of its retained draws, and the same
/// equal-tailed rule at the gate level.
fn rs_credible_level(result: &StudyResult) -> IntervalPair {
    let (lower, upper, level, _) =
        common::reported::response_scalar(result).expect("a published credible interval");
    assert!((level - common::reported::REPORTED_LEVEL).abs() < 1e-12, "level {level}");
    let pair = posterior_pair(result, rs_credible_column(result));
    assert_eq!(
        pair[0].map(|(lo, hi)| (lo.to_bits(), hi.to_bits())),
        Some((lower.to_bits(), upper.to_bits())),
        "the published credible interval [{lower}, {upper}] is not its draws' summaries"
    );
    pair
}

/// Run what the base execution of `spec` needs beyond itself to form its gate-level
/// intervals, on the base execution's data and seed.
fn rs_prepare_gate(spec: &RsSpec, mode: Mode, execution: Variant, data: &TabularData, seed: u64) {
    match (spec.law, spec.query, mode) {
        (RsLaw::Adjustable, RsQuery::Curve, Mode::Frequentist) => {
            // The level enters only the band's quantile.
            let at_gate = rs_execute(
                spec,
                mode,
                execution,
                data,
                (spec.query, &spec.grid),
                common::reported::GATE_LEVEL,
                seed,
            );
            let (lower, upper, level) =
                common::reported::response_band(&at_gate).expect("a gate-level band");
            assert!((level - common::reported::GATE_LEVEL).abs() < 1e-12, "level {level}");
            let gates = lower.into_iter().zip(upper).map(Some).collect();
            rs_gates().lock().unwrap().insert(rs_refit_key(&at_gate), gates);
        }
        (RsLaw::FrontDoor, RsQuery::Curve, Mode::Bayesian) => {
            // One level per execution, on the same structure route, data and seed: its
            // retained draws are those of the curve's grid point when its credible interval
            // reproduces that point's published interval bit for bit.
            for &level in &spec.grid {
                let single = rs_execute(
                    spec,
                    mode,
                    (execution.0, RefuteSuite::None),
                    data,
                    (RsQuery::Level, &[level]),
                    common::reported::REPORTED_LEVEL,
                    seed,
                );
                let [reported, gate] = rs_credible_level(&single);
                let (lower, upper) = reported.expect("a published credible interval");
                rs_gates().lock().unwrap().insert(rs_credible_key(lower, upper), vec![gate]);
            }
        }
        _ => {}
    }
}

/// The published 0.95 pointwise band.
fn rs_band(result: &StudyResult) -> Vec<(f64, f64)> {
    let (lower, upper, level) =
        common::reported::response_band(result).expect("a published pointwise band");
    assert!((level - common::reported::REPORTED_LEVEL).abs() < 1e-12, "band level {level}");
    lower.into_iter().zip(upper).collect()
}

/// `[reported, gate]` per grid point of a Frequentist `functional.effect` band, published as
/// `value ± z·se` per level: the 0.90 interval is `value ± z_0.95·se` with the SE the
/// published half-width states.
fn rs_reformed_band(result: &StudyResult) -> Vec<IntervalPair> {
    let band = rs_band(result);
    let values = rs_values(result);
    assert_eq!(band.len(), values.len(), "one interval per grid point");
    band.into_iter()
        .zip(values)
        .map(|((lo, hi), value)| {
            assert!(
                (0.5 * (lo + hi) - value).abs() <= 1e-9 * (1.0 + value.abs()),
                "the band [{lo}, {hi}] is not centred on the published value {value}"
            );
            let se = (hi - lo) / (2.0 * common::calibration::Z95);
            [Some((lo, hi)), common::reported::normal_at(value, se, common::reported::GATE_LEVEL)]
        })
        .collect()
}

/// `[reported, gate]` per grid point of a Bayesian `functional.effect` band, the gate level
/// from the single-level execution that reproduced each point's interval.
fn rs_credible_band(result: &StudyResult) -> Vec<IntervalPair> {
    let gates = rs_gates().lock().unwrap();
    rs_band(result)
        .into_iter()
        .map(|(lo, hi)| {
            let gate = gates.get(&rs_credible_key(lo, hi)).unwrap_or_else(|| {
                panic!("no single-level execution reproduces the credible interval [{lo}, {hi}]")
            });
            [Some((lo, hi)), gate[0]]
        })
        .collect()
}

/// `[reported, gate]` per grid point of a `response.kennedy_dr` band, the gate level from the
/// refit of the same execution at 0.90.
fn rs_refitted_band(result: &StudyResult) -> Vec<IntervalPair> {
    let gate = rs_gates()
        .lock()
        .unwrap()
        .get(&rs_refit_key(result))
        .cloned()
        .expect("the gate-level refit of this execution");
    let band = rs_band(result);
    assert_eq!(gate.len(), band.len(), "one gate interval per grid point");
    band.into_iter().zip(gate).map(|(reported, gate)| [Some(reported), gate]).collect()
}

fn rs_score(spec: &RsSpec, mode: Mode, result: &StudyResult) -> Vec<IntervalPair> {
    match (spec.law, spec.query, mode) {
        (_, RsQuery::Level, Mode::Frequentist) => {
            vec![common::reported::response_normal_pair(result)]
        }
        (RsLaw::FrontDoor, RsQuery::Level, Mode::Bayesian) => vec![rs_credible_level(result)],
        (RsLaw::FrontDoor, RsQuery::Curve, Mode::Frequentist) => rs_reformed_band(result),
        (RsLaw::FrontDoor, RsQuery::Curve, Mode::Bayesian) => rs_credible_band(result),
        (RsLaw::Adjustable, RsQuery::Curve, Mode::Frequentist) => rs_refitted_band(result),
        (RsLaw::Adjustable, _, Mode::Bayesian) => {
            common::reported::response_posterior_pairs(result)
                .expect("a credible interval with its retained draws")
        }
    }
}

/// Measure one coordinate: `variants[0]` on every replicate, the rest as its siblings.
fn rs_coverage(
    test: &'static str,
    family: u64,
    (law, class, query): (RsLaw, RsClass, RsQuery),
    mode: Mode,
    (record, variants): (BaseRecord, &[Variant]),
) {
    let spec = rs_spec(law, class, query, mode);
    let base = variants[0];
    measure_coordinate_from(
        &Coordinate {
            test,
            dgp: spec.dgp,
            family,
            estimator: spec.estimator,
            interval: spec.interval,
            variants,
        },
        &spec.targets,
        |seed, execution| {
            let data = spec.sample(seed);
            if execution == base {
                rs_prepare_gate(&spec, mode, execution, &data, seed);
            }
            rs_study(
                &spec,
                mode,
                execution,
                data,
                (spec.query, &spec.grid),
                common::reported::REPORTED_LEVEL,
            )
        },
        |result| rs_score(&spec, mode, result),
        record,
    );
}

macro_rules! rs_coverage_tests {
    ($(
        $(#[$attr:meta])*
        $name:ident: $family:literal,
        $(
            $law:ident, $class:ident, $query:ident, $mode:ident,
            $record:ident [$(($via:ident, $suite:ident)),+ $(,)?]
        ),+;
    )+) => {$(
        $(#[$attr])*
        #[test]
        #[ignore = "calibration: run via scripts/measure_calibration.sh"]
        fn $name() {
            sharing_base(|| {$(
                rs_coverage(
                    stringify!($name),
                    $family,
                    (RsLaw::$law, RsClass::$class, RsQuery::$query),
                    Mode::$mode,
                    (BaseRecord::$record, &[$(($via, RefuteSuite::$suite)),+]),
                );
            )+});
        }
    )+};
}

rs_coverage_tests! {
    /// `E[Y | do(T = 1)] = 0.596` over a one-atom ADMG posterior: the atom's
    /// Bayesian-bootstrap credible interval.
    intervention_response_admg_graph_posterior_bayesian_coverage: 0xFE50_0001,
        FrontDoor, Admg, Level, Bayesian, Recorded [
            (Posterior, Cheap), (Posterior, Full),
        ],
        // `E[Y | do(T = 1)] = 0.596` on the given front-door ADMG, Bayesian.
        FrontDoor, Admg, Level, Bayesian, Recorded [
            (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full), (Accepted, Full),
        ];
    /// `E[Y | do(T = 1)] = 0.596` on the given front-door ADMG (general ID,
    /// `functional.effect`).
    intervention_response_admg_given_frequentist_functional_effect_coverage: 0xFE50_0002,
        FrontDoor, Admg, Level, Frequentist, Recorded [
            (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full), (Accepted, Full),
        ],
        // `E[Y | do(T = 1)] = 0.596` of the front-door table over a one-atom ADMG posterior:
        // the atom's plug-in bootstrap-SE interval.
        FrontDoor, Admg, Level, Frequentist, Recorded [
            (Posterior, Cheap), (Posterior, Full),
        ];
    /// `E[Y | do(T = 1)] = 3` on the given ADMG `z -> t -> y` with no bidirected edge
    /// (adjustment, `response.intervention_gcomp`).
    intervention_response_admg_given_frequentist_intervention_gcomp_coverage: 0xFE50_0003,
        Adjustable, Admg, Level, Frequentist, Recorded [
            (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full), (Accepted, Full),
        ],
        // `E[Y | do(T = 1)] = 3` on the given adjustment-identified `TemporalAdmg` lift
        // (`response.intervention_gcomp`).
        Adjustable, TemporalAdmg, Level, Frequentist, Recorded [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    /// `E[Y | do(T = 1)] = 0.596` over a one-atom `TemporalAdmg` posterior, Bayesian.
    intervention_response_temporal_admg_graph_posterior_bayesian_coverage: 0xFE50_0007,
        FrontDoor, TemporalAdmg, Level, Bayesian, Recorded [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ],
        // `E[Y | do(T = 1)] = 0.596` on the given `TemporalAdmg` lift, Bayesian.
        FrontDoor, TemporalAdmg, Level, Bayesian, Recorded [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    /// The front-door curve `[0.314, 0.596]` over a one-atom `TemporalAdmg` posterior,
    /// Bayesian.
    response_curve_temporal_admg_graph_posterior_bayesian_coverage: 0xFE50_0009,
        FrontDoor, TemporalAdmg, Curve, Bayesian, Recorded [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ],
        // The front-door curve `[0.314, 0.596]` on the given `TemporalAdmg` lift, Bayesian.
        FrontDoor, TemporalAdmg, Curve, Bayesian, Recorded [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    /// The front-door curve `[0.314, 0.596]` on the given `TemporalAdmg` lift
    /// (`functional.effect`).
    response_curve_temporal_admg_given_frequentist_functional_effect_coverage: 0xFE50_000A,
        FrontDoor, TemporalAdmg, Curve, Frequentist, Recorded [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ],
        // The front-door curve `[0.314, 0.596]` over a one-atom `TemporalAdmg` posterior.
        FrontDoor, TemporalAdmg, Curve, Frequentist, Recorded [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    /// The curve `1 + 2a` at `a = -1, 0, 1` on the given adjustment-identified `TemporalAdmg`
    /// lift (`response.kennedy_dr`).
    response_curve_temporal_admg_given_frequentist_kennedy_dr_coverage: 0xFE50_000B,
        Adjustable, TemporalAdmg, Curve, Frequentist, Recorded [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ],
        // The curve `1 + 2a` at `a = -1, 0, 1` on the given adjustment-identified ADMG
        // (`response.kennedy_dr`).
        Adjustable, Admg, Curve, Frequentist, Recorded [
            (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full), (Accepted, Full),
        ];
    /// The curve `1 + 2a` at `a = -1, 0, 1` on the given adjustment-identified `TemporalAdmg`
    /// lift (`response.bayesian`).
    response_curve_temporal_admg_given_bayesian_response_bayesian_coverage: 0xFE50_000D,
        Adjustable, TemporalAdmg, Curve, Bayesian, Recorded [
            (Explicit, None), (Accepted, None),
        ];
    /// `E[Y | do(T = 1)] = 0.596` on the given `TemporalAdmg` lift (`functional.effect`).
    intervention_response_temporal_admg_given_frequentist_functional_effect_coverage: 0xFE50_000E,
        FrontDoor, TemporalAdmg, Level, Frequentist, Recorded [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ],
        // `E[Y | do(T = 1)] = 0.596` over a one-atom `TemporalAdmg` posterior on the table.
        FrontDoor, TemporalAdmg, Level, Frequentist, Recorded [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    /// `E[Y | do(T = 1)] = 3` on the given adjustment-identified `TemporalAdmg` lift
    /// (`response.bayesian`).
    intervention_response_temporal_admg_given_bayesian_response_bayesian_coverage: 0xFE50_0011,
        Adjustable, TemporalAdmg, Level, Bayesian, Recorded [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ],
        // `E[Y | do(T = 1)] = 3` on the given adjustment-identified ADMG (`response.bayesian`).
        Adjustable, Admg, Level, Bayesian, Recorded [
            (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full), (Accepted, Full),
        ];
    /// The front-door curve `[0.314, 0.596]` over a one-atom ADMG posterior.
    response_curve_admg_graph_posterior_frequentist_coverage: 0xFE50_0012,
        FrontDoor, Admg, Curve, Frequentist, Unrecorded [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ],
        // The front-door curve `[0.314, 0.596]` on the given front-door ADMG
        // (`functional.effect`).
        FrontDoor, Admg, Curve, Frequentist, Unrecorded [
            (Explicit, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
            (Accepted, Full),
        ];
    /// The front-door curve `[0.314, 0.596]` over a one-atom ADMG posterior, Bayesian.
    response_curve_admg_graph_posterior_bayesian_coverage: 0xFE50_0013,
        FrontDoor, Admg, Curve, Bayesian, Unrecorded [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ],
        // The front-door curve `[0.314, 0.596]` on the given front-door ADMG, Bayesian.
        FrontDoor, Admg, Curve, Bayesian, Unrecorded [
            (Explicit, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
            (Accepted, Full),
        ];
}
// ---------------------------------------------------------------- temporal window
//
// The pulse, sustained and conditional contrasts and the natural mediation decomposition
// that read the unrolled window of a stationary template: a `TemporalAdmg` (given, accepted
// or as a one-atom graph posterior of lagged `TemporalAdmg` atoms) and a `TemporalPag`
// (given, accepted or as a one-atom graph posterior of lagged `Pag` atoms).
//
// The laws are the two binary stationary structural processes of
// `conformance/estimate/temporal_window_truth` (`reference.py`), every noise uniform on
// equally likely bins. The rows of a window table are overlapping windows of one dependent
// series, so they are not iid draws of a finite population: each replicate simulates the
// structural process itself, started in its stationary law, and the population value of
// each reported functional is the fixture's exact enumerated truth.
//
// The window functional is a plug-in on the lag-aligned window table. Its Frequentist
// interval is `estimate ± z·se` from the circular-block bootstrap of consecutive windows
// (199 replicates; the estimate records the row-block family, so the runtime reports
// `circular_block_se`), re-formed at 0.90 from the same SE. Its Bayesian interval is the
// equal-tailed quantile interval of the Bayesian-bootstrap draws (one Dirichlet weight per
// block), re-summarized at 0.90 from those draws. A one-atom posterior keeps the atom's own
// uncertainty and posterior, so it reports the same two intervals.

/// `r(x, w, y(t-1))` of the contrast process, indexed `[y(t-1)][x][w]`: `y(t) = 1` when a
/// draw uniform on four bins falls below it.
const TW_CONTRAST_THRESHOLD: [[[usize; 2]; 2]; 2] = [[[0, 1], [2, 1]], [[2, 3], [4, 3]]];

/// `q(x, w)` of the mediation process, indexed `[x][w]`: `m = 1` when a draw uniform on four
/// bins falls below it.
const TW_MEDIATOR_THRESHOLD: [[usize; 2]; 2] = [[1, 2], [2, 3]];

/// `s(x, m)` of the mediation process, indexed `[x][m]`: `y = 1` when a draw uniform on two
/// bins falls below it.
const TW_OUTCOME_THRESHOLD: [[usize; 2]; 2] = [[0, 1], [1, 1]];

/// Base series length of the window contrasts; `grid_n` measures `n/2, n, 2n`.
///
/// The contrast is identified by the complete ID algorithm on the three-period window, whose
/// factors condition on every earlier coordinate of the window: the outcome factor reads the
/// 8-coordinate history `(x, w, y)(0), (x, w, y)(1), (x, w)(2)`, whose rarest cell of positive
/// probability has probability `1/1024`. The plug-in is evaluable only when every such history
/// that follows an observed prefix is itself observed (a zero-weight prefix contributes
/// nothing), which fails on a series of length `L` with probability about
/// `Σ 4·(exp(-L·p/4) - exp(-L·p))` over the prefix cells `p` (2.4e-3 at 9000 windows, 2.6e-6
/// at 16000; a simulation of the process agrees). The validated siblings refit the contrast on
/// a contiguous half of the windows (`window.stretch_50`, `window.split_half`), so the
/// smallest grid point must leave 16000 windows in a half: `n/2 = 32000`.
const TW_CONTRAST_BASE_N: usize = 64_000;

/// Base series length of the mediation decomposition; `grid_n` measures `n/2, n, 2n`.
///
/// The edge g-formula reads `P(w)`, `P(m | x, w)` and `P(y | x, m)` at the pulse period: the
/// rarest conditioning cell `(x, m) = (0, 1)` has probability `3/16`, so every cell is present
/// at the smallest grid point and in every half-sample refit. The size is the ~1000 rows the
/// tabular `functional.effect` coverage designs use.
const TW_MEDIATION_BASE_N: usize = 1_000;

/// Fixture of the exact enumerated window truths.
fn tw_fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_window_truth/expected.json"
    ))
    .unwrap()
}

/// `n` periods of `x, y, w` from the contrast process: `x`, `w` fair coins,
/// `y(t) = 1[U(t) < r(x(t), w(t), y(t-1))]` with `U` uniform on four bins.
///
/// The mean of `r` over `(x, w)` is 1 after `y = 0` and 3 after `y = 1`, so `y` is a Markov
/// chain with `P(1 | 0) = 1/4`, `P(1 | 1) = 3/4` and stationary law `P(y = 1) = 1/2`. The
/// period before the series draws `y` from that law (a fair coin), so the series is stationary
/// from its first period, as the fixture's window law is.
fn tw_contrast_series(n: usize, seed: u64) -> antecedent_data::TimeSeriesData {
    let mut u01 = uniform(seed, 0xFE60_C0A7);
    let mut bin = |bins: usize| ((u01() * bins as f64) as usize).min(bins - 1);
    let mut y_prev = bin(2);
    let (mut x, mut y, mut w) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let (xt, wt, uy) = (bin(2), bin(2), bin(4));
        let yt = usize::from(uy < TW_CONTRAST_THRESHOLD[y_prev][xt][wt]);
        x.push(xt as f64);
        y.push(yt as f64);
        w.push(wt as f64);
        y_prev = yt;
    }
    antecedent_data::TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("y", y.as_slice()), ("w", w.as_slice())],
        1,
    )
    .unwrap()
}

/// `n` periods of `c, x, w, m, y` from the mediation process: `c`, `w` fair coins,
/// `x(t) = 1[Ux < 1 + 2 c(t-1)]` (`Ux` on four bins), `m = 1[Um < q(x, w)]` (`Um` on four
/// bins), `y = 1[Uy < s(x, m)]` (`Uy` on two bins).
///
/// `c` is iid, so drawing the period before the series as a fair coin starts the process in
/// its stationary law.
fn tw_mediation_series(n: usize, seed: u64) -> antecedent_data::TimeSeriesData {
    let mut u01 = uniform(seed, 0xFE60_3ED1);
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

/// The generating template of the contrast process (`x = 0`, `y = 1`, `w = 2`):
/// `x -> y`, `w -> y` within a period and `y(t-1) -> y(t)`.
fn tw_contrast_template() -> TemporalAdmg {
    use antecedent_core::Lag;
    let mut graph = TemporalAdmg::empty();
    let y1 = graph.add_lagged(v(1), Lag::from_raw(1)).unwrap();
    let x0 = graph.add_lagged(v(0), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = graph.add_lagged(v(2), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(v(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(y1, y0).unwrap();
    graph.insert_directed(w0, y0).unwrap();
    graph.insert_directed(x0, y0).unwrap();
    graph
}

/// The generating template of the mediation process (`c = 0`, `x = 1`, `w = 2`, `m = 3`,
/// `y = 4`): `c(t-1) -> x`, `x -> m`, `w -> m`, `m -> y`, `x -> y`.
fn tw_mediation_admg() -> TemporalAdmg {
    use antecedent_core::Lag;
    let mut graph = TemporalAdmg::empty();
    let c1 = graph.add_lagged(v(0), Lag::from_raw(1)).unwrap();
    let x0 = graph.add_lagged(v(1), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = graph.add_lagged(v(2), Lag::CONTEMPORANEOUS).unwrap();
    let m0 = graph.add_lagged(v(3), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(v(4), Lag::CONTEMPORANEOUS).unwrap();
    for (from, to) in [(c1, x0), (x0, m0), (w0, m0), (m0, y0), (x0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

/// The mediation template as a `TemporalPag`: every edge into `m` and `y` and out of `x` is
/// visible (`c(t-1)` points into `x`, `w` into `m`), so the class holds one functional.
fn tw_mediation_pag() -> TemporalPag {
    use antecedent_core::Lag;
    let mut graph = TemporalPag::empty();
    let c1 = graph.add_lagged(v(0), Lag::from_raw(1)).unwrap();
    let x0 = graph.add_lagged(v(1), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = graph.add_lagged(v(2), Lag::CONTEMPORANEOUS).unwrap();
    let m0 = graph.add_lagged(v(3), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(v(4), Lag::CONTEMPORANEOUS).unwrap();
    for (from, to) in [(c1, x0), (x0, m0), (w0, m0), (m0, y0), (x0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

/// A graph posterior holding all its mass on one stationary lagged template: the
/// contemporaneous edges and the lag-1 directed edges `(lag, from, to)`, as a lagged
/// `TemporalAdmg` atom (no lagged latent confounder) or a lagged `Pag` atom.
fn tw_lagged_posterior(
    nodes: usize,
    kind: GraphPosteriorAtomKind,
    contemporaneous: &[(u32, u32)],
    lagged: &[(u32, usize, usize)],
) -> GraphPosterior {
    let lag_mask =
        lagged.iter().fold(antecedent_discovery::LagMask::EMPTY, |word, &(lag, from, to)| {
            antecedent_discovery::with_lagged_edge(word, nodes, lag, from, to, true).unwrap()
        });
    let posterior = GraphPosterior::from_atoms(
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
    .with_atom_kind(kind)
    .with_lagged_marginals(1, vec![0.0; nodes * nodes])
    .unwrap()
    .with_lag_masks(vec![lag_mask])
    .unwrap();
    match kind {
        GraphPosteriorAtomKind::TemporalAdmg => {
            posterior.with_lag_bidirected_masks(vec![antecedent_discovery::LagMask::EMPTY]).unwrap()
        }
        _ => posterior,
    }
}

/// The analyses of this family, each with its law, graph, query and truth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TwDesign {
    /// `do(x(1))` read at `y(2)`: truth `1/8`. The treatment moves `P(y(1) = 1)` from
    /// `(1/8 + 5/8)/2 = 3/8` to `(3/8 + 7/8)/2 = 5/8`, and `y(2)` follows `y(1)` with
    /// `P(1 | 1) - P(1 | 0) = 3/4 - 1/4`, so the contrast is `1/4 · 1/2 = 1/8`.
    Pulse,
    /// `do(x(1), x(2))` read at `y(2)`: truth `3/8`. Under the active policy
    /// `E[y(2)] = 3/8 · 3/8 + 5/8 · 7/8 = 44/64`; under the control policy
    /// `5/8 · 1/8 + 3/8 · 5/8 = 20/64`; the contrast is `24/64 = 3/8`.
    Sustained,
    /// The pulse conditioned on `w(1)`, which no treated period reaches: `w` is independent of
    /// the treatment and of the outcome's history, so the averaged conditional contrast is
    /// the pulse's `1/8`.
    ConditionalPulse,
    /// The sustained contrast conditioned on `w(1)`: `3/8`, as for the conditional pulse.
    ConditionalSustained,
    /// The pure natural indirect effect `E[Y(0, M(1))] - E[Y(0, M(0))]` of `x` on `y` through
    /// `m` at the pulse period, on the `TemporalAdmg`: truth `1/8`. Under `x = 0` the outcome
    /// is `1` with probability `m/2`, and `P(M(1) = 1) = (2 + 3)/8 = 5/8`,
    /// `P(M(0) = 1) = (1 + 2)/8 = 3/8`, so the effect is `(5/8 - 3/8)/2 = 1/8`.
    MediationAdmg,
    /// The same natural indirect effect on the `TemporalPag` of the same template: `1/8`.
    MediationPag,
}

impl TwDesign {
    /// The fixture's exact enumerated truth, checked against its closed form.
    fn truth(self) -> f64 {
        let fixture = tw_fixture();
        let (value, exact) = match self {
            Self::Pulse => (&fixture["contrast"]["truth"]["pulse"], 0.125),
            Self::Sustained => (&fixture["contrast"]["truth"]["sustained"], 0.375),
            Self::ConditionalPulse => (&fixture["contrast"]["truth"]["conditional_pulse"], 0.125),
            Self::ConditionalSustained => {
                (&fixture["contrast"]["truth"]["conditional_sustained"], 0.375)
            }
            Self::MediationAdmg | Self::MediationPag => {
                (&fixture["mediation"]["truth"]["natural_indirect_effect"], 0.125)
            }
        };
        let value = value.as_f64().unwrap();
        assert!((value - exact).abs() < 1e-15, "{self:?}: fixture truth {value}");
        value
    }

    fn is_contrast(self) -> bool {
        matches!(
            self,
            Self::Pulse | Self::Sustained | Self::ConditionalPulse | Self::ConditionalSustained
        )
    }

    fn dgp(self) -> &'static str {
        if self.is_contrast() {
            "temporal_window_contrast_process"
        } else {
            "temporal_window_mediation_process"
        }
    }

    /// The replicate's series, which depends on `seed` only.
    fn series(self, seed: u64) -> antecedent_data::TimeSeriesData {
        if self.is_contrast() {
            tw_contrast_series(common::calibration::grid_n(TW_CONTRAST_BASE_N), seed)
        } else {
            tw_mediation_series(common::calibration::grid_n(TW_MEDIATION_BASE_N), seed)
        }
    }

    /// The window query: the pulse at the first treated period read at the next period, the
    /// sustained policy on that period and the one after it, each conditioned on `w` at the
    /// first treated period; the natural indirect effect of `x` on `y` through `m`.
    fn query(self) -> antecedent_core::CausalQuery {
        use antecedent_core::{
            CausalQuery, MediationContrast, MediationQuery, TemporalEffectQuery, TemporalPolicy,
        };
        let pulse = TemporalEffectQuery::pulse(v(0), v(1), 1.0).with_horizon_steps(2);
        let sustained = || pulse.clone().with_policy(TemporalPolicy::sustained(0, 1));
        let contrast = match self {
            Self::Pulse => pulse.clone(),
            Self::Sustained => sustained(),
            Self::ConditionalPulse => pulse.clone().with_effect_modifiers([v(2)]),
            Self::ConditionalSustained => sustained().with_effect_modifiers([v(2)]),
            Self::MediationAdmg | Self::MediationPag => {
                return CausalQuery::Mediation(MediationQuery::binary(
                    v(1),
                    v(4),
                    [v(3)],
                    MediationContrast::NaturalIndirect,
                ));
            }
        };
        CausalQuery::TemporalEffect(contrast)
    }

    /// The one-atom graph posterior of the design's template.
    fn posterior(self) -> GraphPosterior {
        let mediation = [(1, 3), (2, 3), (3, 4), (1, 4)];
        match self {
            Self::MediationAdmg => tw_lagged_posterior(
                5,
                GraphPosteriorAtomKind::TemporalAdmg,
                &mediation,
                &[(1, 0, 1)],
            ),
            Self::MediationPag => {
                tw_lagged_posterior(5, GraphPosteriorAtomKind::Pag, &mediation, &[(1, 0, 1)])
            }
            _ => tw_lagged_posterior(
                3,
                GraphPosteriorAtomKind::TemporalAdmg,
                &[(0, 1), (2, 1)],
                &[(1, 1, 1)],
            ),
        }
    }

    fn attach(self, builder: StudyBuilder, via: Via) -> StudyBuilder {
        match (self, via) {
            (_, Posterior) => builder.graph_posterior(self.posterior()),
            (Self::MediationPag, Explicit) => builder.graph(tw_mediation_pag()),
            (Self::MediationPag, Accepted) => {
                builder.graph(AcceptedGraph::from(tw_mediation_pag()))
            }
            (Self::MediationAdmg, Explicit) => builder.graph(tw_mediation_admg()),
            (Self::MediationAdmg, Accepted) => {
                builder.graph(AcceptedGraph::from(tw_mediation_admg()))
            }
            (_, Explicit) => builder.graph(tw_contrast_template()),
            (_, Accepted) => builder.graph(AcceptedGraph::from(tw_contrast_template())),
        }
    }

    fn study(self, mode: Mode, seed: u64, (via, suite): Variant) -> Study {
        self.attach(Study::series(self.series(seed)), via)
            .query(self.query())
            .inference(mode.inference())
            .refute(suite)
            .build()
            .unwrap()
    }
}

/// A given template, explicit and accepted, under every validation mode.
const TW_GIVEN: &[Variant] = &[
    (Explicit, RefuteSuite::None),
    (Explicit, RefuteSuite::Cheap),
    (Explicit, RefuteSuite::Full),
    (Accepted, RefuteSuite::None),
    (Accepted, RefuteSuite::Cheap),
    (Accepted, RefuteSuite::Full),
];

/// A one-atom graph posterior under every validation mode.
const TW_POSTERIOR: &[Variant] = &[
    (Posterior, RefuteSuite::None),
    (Posterior, RefuteSuite::Cheap),
    (Posterior, RefuteSuite::Full),
];

/// `IntervalMethod::as_str()` the window functional reports: the circular-block bootstrap SE
/// (the estimate records the row-block family) or the posterior quantiles.
fn tw_interval(mode: Mode) -> &'static str {
    match mode {
        Mode::Frequentist => "circular_block_se",
        Mode::Bayesian => mode.interval(),
    }
}

/// Measure one coordinate of the family: the design's scalar against its exact truth.
fn tw_measure(test: &'static str, family: u64, mode: Mode, design: TwDesign, variants: &[Variant]) {
    let targets = Target::scalar(design.truth());
    measure_coordinate(
        &Coordinate {
            test,
            dgp: design.dgp(),
            family,
            estimator: FUNCTIONAL,
            interval: tw_interval(mode),
            variants,
        },
        &targets,
        move |seed, variant| design.study(mode, seed, variant),
        move |result| vec![effect_pair(mode, result)],
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_admg_given_frequentist_coverage() {
    sharing_base(|| {
        {
            tw_measure(
                "sustained_effect_temporal_admg_given_frequentist_coverage",
                0xFE60_0002,
                Mode::Frequentist,
                TwDesign::Sustained,
                TW_GIVEN,
            );
        }
        {
            tw_measure(
                "sustained_effect_temporal_admg_given_frequentist_coverage",
                0xFE60_0002,
                Mode::Frequentist,
                TwDesign::Sustained,
                TW_POSTERIOR,
            );
        }
    });
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_admg_given_bayesian_coverage() {
    sharing_base(|| {
        {
            tw_measure(
                "sustained_effect_temporal_admg_given_bayesian_coverage",
                0xFE60_0003,
                Mode::Bayesian,
                TwDesign::Sustained,
                TW_GIVEN,
            );
        }
        {
            tw_measure(
                "sustained_effect_temporal_admg_given_bayesian_coverage",
                0xFE60_0003,
                Mode::Bayesian,
                TwDesign::Sustained,
                TW_POSTERIOR,
            );
        }
    });
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_pulse_temporal_admg_given_bayesian_coverage() {
    sharing_base(|| {
        {
            tw_measure(
                "temporal_conditional_pulse_temporal_admg_given_bayesian_coverage",
                0xFE60_0005,
                Mode::Bayesian,
                TwDesign::ConditionalPulse,
                TW_GIVEN,
            );
        }
        {
            tw_measure(
                "temporal_conditional_pulse_temporal_admg_given_bayesian_coverage",
                0xFE60_0005,
                Mode::Bayesian,
                TwDesign::ConditionalPulse,
                TW_POSTERIOR,
            );
        }
    });
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_sustained_temporal_admg_given_bayesian_coverage() {
    sharing_base(|| {
        {
            tw_measure(
                "temporal_conditional_sustained_temporal_admg_given_bayesian_coverage",
                0xFE60_0007,
                Mode::Bayesian,
                TwDesign::ConditionalSustained,
                TW_GIVEN,
            );
        }
        {
            tw_measure(
                "temporal_conditional_sustained_temporal_admg_given_bayesian_coverage",
                0xFE60_0007,
                Mode::Bayesian,
                TwDesign::ConditionalSustained,
                TW_POSTERIOR,
            );
        }
    });
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_admg_graph_posterior_frequentist_coverage() {
    sharing_base(|| {
        {
            tw_measure(
                "pulse_effect_temporal_admg_graph_posterior_frequentist_coverage",
                0xFE60_0008,
                Mode::Frequentist,
                TwDesign::Pulse,
                TW_POSTERIOR,
            );
        }
        {
            tw_measure(
                "pulse_effect_temporal_admg_graph_posterior_frequentist_coverage",
                0xFE60_0008,
                Mode::Frequentist,
                TwDesign::Pulse,
                TW_GIVEN,
            );
        }
    });
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_admg_graph_posterior_bayesian_coverage() {
    sharing_base(|| {
        {
            tw_measure(
                "pulse_effect_temporal_admg_graph_posterior_bayesian_coverage",
                0xFE60_0009,
                Mode::Bayesian,
                TwDesign::Pulse,
                TW_POSTERIOR,
            );
        }
        {
            tw_measure(
                "pulse_effect_temporal_admg_graph_posterior_bayesian_coverage",
                0xFE60_0009,
                Mode::Bayesian,
                TwDesign::Pulse,
                TW_GIVEN,
            );
        }
    });
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_pulse_temporal_admg_graph_posterior_frequentist_coverage() {
    sharing_base(|| {
        {
            tw_measure(
                "temporal_conditional_pulse_temporal_admg_graph_posterior_frequentist_coverage",
                0xFE60_000C,
                Mode::Frequentist,
                TwDesign::ConditionalPulse,
                TW_POSTERIOR,
            );
        }
        {
            tw_measure(
                "temporal_conditional_pulse_temporal_admg_graph_posterior_frequentist_coverage",
                0xFE60_000C,
                Mode::Frequentist,
                TwDesign::ConditionalPulse,
                TW_GIVEN,
            );
        }
    });
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_sustained_temporal_admg_graph_posterior_frequentist_coverage() {
    sharing_base(|| {
        {
            tw_measure(
                "temporal_conditional_sustained_temporal_admg_graph_posterior_frequentist_coverage",
                0xFE60_000E,
                Mode::Frequentist,
                TwDesign::ConditionalSustained,
                TW_POSTERIOR,
            );
        }
        {
            tw_measure(
                "temporal_conditional_sustained_temporal_admg_graph_posterior_frequentist_coverage",
                0xFE60_000E,
                Mode::Frequentist,
                TwDesign::ConditionalSustained,
                TW_GIVEN,
            );
        }
    });
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_mediation_effect_temporal_admg_given_frequentist_coverage() {
    sharing_base(|| {
        {
            tw_measure(
                "temporal_mediation_effect_temporal_admg_given_frequentist_coverage",
                0xFE60_0010,
                Mode::Frequentist,
                TwDesign::MediationAdmg,
                TW_GIVEN,
            );
        }
        {
            tw_measure(
                "temporal_mediation_effect_temporal_admg_given_frequentist_coverage",
                0xFE60_0010,
                Mode::Frequentist,
                TwDesign::MediationAdmg,
                TW_POSTERIOR,
            );
        }
    });
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_mediation_effect_temporal_admg_given_bayesian_coverage() {
    sharing_base(|| {
        {
            tw_measure(
                "temporal_mediation_effect_temporal_admg_given_bayesian_coverage",
                0xFE60_0011,
                Mode::Bayesian,
                TwDesign::MediationAdmg,
                TW_GIVEN,
            );
        }
        {
            tw_measure(
                "temporal_mediation_effect_temporal_admg_given_bayesian_coverage",
                0xFE60_0011,
                Mode::Bayesian,
                TwDesign::MediationAdmg,
                TW_POSTERIOR,
            );
        }
    });
}

/// The natural indirect effect on the window of the mediation template's `TemporalPag`.
///
/// The unrolled window's MAG completions are read through every ADMG sharing their
/// m-separation model, and those identify the components by different functionals, so the
/// execution takes the class route (`estimate_pag_class_components`) and publishes the class
/// identified set: Imbens-Manski over circular-block shared replicates at 0.95 (Frequentist),
/// the product-posterior envelope at 0.90 (Bayesian). The generating template is one of the
/// candidates and the confounding the others add touches only `c(t-1) -> x` and `w -> m`, so
/// every candidate takes the natural indirect effect 1/8 on the stationary law.
fn tw_identified_set(test: &'static str, family: u64, mode: Mode) {
    let design = TwDesign::MediationPag;
    measure_identified_set(
        &SetCoordinate {
            test,
            dgp: design.dgp(),
            family,
            mode,
            variants: TW_GIVEN,
            target: "the population identified set, the point at the natural indirect effect \
                     every candidate functional takes on the stationary window law",
        },
        design.truth(),
        move |seed, variant| design.study(mode, seed, variant),
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_mediation_effect_temporal_pag_given_frequentist_coverage() {
    tw_identified_set(
        "temporal_mediation_effect_temporal_pag_given_frequentist_coverage",
        0xFE60_0014,
        Mode::Frequentist,
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_mediation_effect_temporal_pag_given_bayesian_coverage() {
    tw_identified_set(
        "temporal_mediation_effect_temporal_pag_given_bayesian_coverage",
        0xFE60_0015,
        Mode::Bayesian,
    );
}

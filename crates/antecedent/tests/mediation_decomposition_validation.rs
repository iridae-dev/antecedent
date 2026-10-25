//! Cheap and full validation of a mediation decomposition over a mixed graph, a graph class, a
//! temporal window and a posterior over lagged templates.
//!
//! The total, natural direct and natural indirect effects are three functionals of one table, so
//! the suite refits all of them together: the sensitivity of the indirect effect to a violation
//! of sequential ignorability, a mediator placebo, and the joint stability of the components on
//! the same perturbed tables. The references are computed here from the contingency tables and
//! from the generating equations: the textbook mediation sensitivity of Imai, Keele and Yamamoto
//! for the working linear model of the table, the counted decomposition, and the structural
//! effects of the processes that generated the series.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    reason = "test fixtures: small exact indices, levels and enumerated probabilities"
)]

use antecedent::{BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, MediationContrast, MediationQuery, TemporalEffectQuery,
    VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_discovery::adjacency_mask::set_edge;
use antecedent_discovery::{
    AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, TemporalAdmgAtom,
    with_lagged_edge,
};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Pag, TemporalAdmg, TemporalPag};
use antecedent_prob::InferenceDiagnostics;

type Cells = [(f64, f64, f64, usize)];

fn var(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn node(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

fn bern(p: f64, level: usize) -> f64 {
    if level == 1 { p } else { 1.0 - p }
}

/// `(t, m, y, count)` of the table every static test below counts from.
const CELLS: [(f64, f64, f64, usize); 8] = [
    (0.0, 0.0, 0.0, 30),
    (0.0, 0.0, 1.0, 10),
    (0.0, 1.0, 0.0, 5),
    (0.0, 1.0, 1.0, 15),
    (1.0, 0.0, 0.0, 12),
    (1.0, 0.0, 1.0, 8),
    (1.0, 1.0, 0.0, 20),
    (1.0, 1.0, 1.0, 25),
];

fn columns_of(cells: &Cells) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut t, mut m, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for &(tv, mv, yv, count) in cells {
        for _ in 0..count {
            t.push(tv);
            m.push(mv);
            y.push(yv);
        }
    }
    (t, m, y)
}

fn table_of(cells: &Cells) -> TabularData {
    let (t, m, y) = columns_of(cells);
    TabularData::from_f64_columns([("t", t.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

fn count(cells: &Cells, t: Option<f64>, m: Option<f64>, y: Option<f64>) -> f64 {
    cells
        .iter()
        .filter(|(tv, mv, yv, _)| {
            t.is_none_or(|v| v == *tv) && m.is_none_or(|v| v == *mv) && y.is_none_or(|v| v == *yv)
        })
        .map(|(_, _, _, n)| *n as f64)
        .sum()
}

fn e_y(cells: &Cells, t: f64, m: f64) -> f64 {
    count(cells, Some(t), Some(m), Some(1.0)) / count(cells, Some(t), Some(m), None)
}

fn p_m_cells(cells: &Cells, m: f64, t: f64) -> f64 {
    count(cells, Some(t), Some(m), None) / count(cells, Some(t), None, None)
}

fn counted_total(cells: &Cells) -> f64 {
    count(cells, Some(1.0), None, Some(1.0)) / count(cells, Some(1.0), None, None)
        - count(cells, Some(0.0), None, Some(1.0)) / count(cells, Some(0.0), None, None)
}

fn counted_direct(cells: &Cells) -> f64 {
    [0.0, 1.0]
        .into_iter()
        .map(|m| (e_y(cells, 1.0, m) - e_y(cells, 0.0, m)) * p_m_cells(cells, m, 0.0))
        .sum()
}

fn counted_indirect(cells: &Cells) -> f64 {
    [0.0, 1.0]
        .into_iter()
        .map(|m| e_y(cells, 0.0, m) * (p_m_cells(cells, m, 1.0) - p_m_cells(cells, m, 0.0)))
        .sum()
}

/// The mediation sensitivity of the working linear model `M ~ T`, `Y ~ T` of a table with no
/// covariate, computed from the cells (Imai, Keele and Yamamoto 2010, Theorem 2).
struct Sensitivity {
    rho_tilde: f64,
    beta_mediator: f64,
    sigma_total: f64,
    sigma_mediator: f64,
}

impl Sensitivity {
    fn of(cells: &Cells) -> Self {
        let n: f64 = cells.iter().map(|c| c.3 as f64).sum();
        let mean = |t: f64, pick: fn(&(f64, f64, f64, usize)) -> f64| -> f64 {
            let rows: f64 = cells.iter().filter(|c| c.0 == t).map(|c| c.3 as f64).sum();
            cells.iter().filter(|c| c.0 == t).map(|c| pick(c) * c.3 as f64).sum::<f64>() / rows
        };
        let (m_mean, y_mean) =
            ([mean(0.0, |c| c.1), mean(1.0, |c| c.1)], [mean(0.0, |c| c.2), mean(1.0, |c| c.2)]);
        let (mut rss_y, mut rss_m, mut cross) = (0.0, 0.0, 0.0);
        for &(t, m, y, rows) in cells {
            let arm = t as usize;
            let (ry, rm) = (y - y_mean[arm], m - m_mean[arm]);
            rss_y += rows as f64 * ry * ry;
            rss_m += rows as f64 * rm * rm;
            cross += rows as f64 * ry * rm;
        }
        let dof = n - 2.0;
        Self {
            rho_tilde: cross / (rss_y.sqrt() * rss_m.sqrt()),
            beta_mediator: m_mean[1] - m_mean[0],
            sigma_total: (rss_y / dof).sqrt(),
            sigma_mediator: (rss_m / dof).sqrt(),
        }
    }

    fn acme(&self, rho: f64) -> f64 {
        let scale = self.beta_mediator * self.sigma_total / self.sigma_mediator;
        scale * (self.rho_tilde - rho * ((1.0 - self.rho_tilde.powi(2)) / (1.0 - rho * rho)).sqrt())
    }

    /// The ACME at the end of the probed range `rho = +-0.2` that is closer to zero, and whether
    /// the effect keeps its sign over the whole range.
    fn worst_and_survives(&self) -> (f64, bool) {
        let (centre, low, high) = (self.acme(0.0), self.acme(-0.2), self.acme(0.2));
        let sign = centre.signum();
        let worst = if sign * low < sign * high { low } else { high };
        (worst, centre != 0.0 && sign * low > 0.0 && sign * high > 0.0)
    }
}

fn mediation_query(contrast: MediationContrast) -> MediationQuery {
    MediationQuery::binary(var(0), var(2), [var(1)], contrast)
}

fn latent_free() -> Admg {
    let mut admg = Admg::with_variables(3);
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        admg.insert_directed(node(a), node(b)).unwrap();
    }
    admg
}

fn study_on(
    data: TabularData,
    graph: impl antecedent::IntoGraphInput,
    contrast: MediationContrast,
    inference: InferenceMode,
    suite: RefuteSuite,
) -> Study {
    Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Mediation(mediation_query(contrast)))
        .inference(inference)
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200))
}

fn ids(result: &StudyResult) -> Vec<String> {
    result.refutations.iter().map(|report| report.refuter.to_string()).collect()
}

fn report<'a>(result: &'a StudyResult, id: &str) -> &'a antecedent_validate::RefutationReport {
    result
        .refutations
        .iter()
        .find(|report| report.refuter.as_ref() == id)
        .unwrap_or_else(|| panic!("no report {id} in {:?}", ids(result)))
}

const CHEAP_IDS: [&str; 8] = [
    "mediation.sequential_ignorability",
    "mediation.placebo_mediator_total",
    "mediation.placebo_mediator_direct",
    "mediation.placebo_mediator_indirect",
    "mediation.subset_total",
    "mediation.subset_direct",
    "mediation.subset_indirect",
    "mediation.joint_stability_subset",
];

const FULL_ONLY_IDS: [&str; 5] = [
    "mediation.half_sample_total",
    "mediation.half_sample_direct",
    "mediation.half_sample_indirect",
    "mediation.joint_stability_half_sample",
    "mediation.joint_bootstrap_range",
];

fn assert_suite(result: &StudyResult, suite: RefuteSuite) {
    let found = ids(result);
    match suite {
        RefuteSuite::None => assert!(found.is_empty(), "{found:?}"),
        RefuteSuite::Full => {
            assert_eq!(found.len(), CHEAP_IDS.len() + FULL_ONLY_IDS.len(), "{found:?}");
            for id in CHEAP_IDS.iter().chain(&FULL_ONLY_IDS) {
                assert!(found.iter().any(|f| f == id), "missing {id} in {found:?}");
            }
        }
        _ => {
            assert_eq!(found.len(), CHEAP_IDS.len(), "{found:?}");
            for id in CHEAP_IDS {
                assert!(found.iter().any(|f| f == id), "missing {id} in {found:?}");
            }
        }
    }
}

#[test]
fn an_admg_decomposition_runs_the_mediation_suite_on_both_inferences() {
    let ctx = ExecutionContext::for_tests(201);
    for inference in [InferenceMode::Frequentist, bayesian()] {
        let none = study_on(
            table_of(&CELLS),
            latent_free(),
            MediationContrast::NaturalIndirect,
            inference.clone(),
            RefuteSuite::None,
        )
        .run(&ctx)
        .unwrap();
        assert_suite(&none, RefuteSuite::None);
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let built = study_on(
                table_of(&CELLS),
                latent_free(),
                MediationContrast::NaturalIndirect,
                inference.clone(),
                suite,
            );
            // The state executes and is never a licence.
            assert_eq!(built.support_status(), Some(CellStatus::Refused));
            let result = built.run(&ctx).unwrap_or_else(|e| panic!("{suite:?}: {e}"));
            assert_suite(&result, suite);
            // Validation observes the estimate, it does not move it.
            assert!((result.effect() - none.effect()).abs() < 1e-12);
            let components = result.payload.mediation().expect("the decomposition");
            assert!((components.total.unwrap() - counted_total(&CELLS)).abs() < 0.05);
            // The components are refitted together on the same perturbed tables.
            assert!(report(&result, "mediation.subset_total").replicates >= 18);
            assert!(report(&result, "mediation.joint_stability_subset").replicates >= 18);
        }
    }
}

#[test]
fn the_sensitivity_report_is_the_textbook_sensitivity_of_the_counted_table() {
    let ctx = ExecutionContext::for_tests(202);
    let reference = Sensitivity::of(&CELLS);
    let (worst, survives) = reference.worst_and_survives();
    for inference in [InferenceMode::Frequentist, bayesian()] {
        let result = study_on(
            table_of(&CELLS),
            latent_free(),
            MediationContrast::NaturalIndirect,
            inference,
            RefuteSuite::Cheap,
        )
        .run(&ctx)
        .unwrap();
        let sensitivity = report(&result, "mediation.sequential_ignorability");
        // The break-even correlation and the curve at the end of the probed range are the
        // closed form of the paper, computed here from the cells.
        assert!((sensitivity.comparison - reference.rho_tilde).abs() < 1e-9, "{sensitivity:?}");
        assert!((sensitivity.refuted_ate - worst).abs() < 1e-9, "{sensitivity:?}");
        assert_eq!(sensitivity.passed, survives);
        assert!(sensitivity.informative);
    }
}

#[test]
fn a_mediator_that_explains_the_outcome_survives_the_sensitivity_and_a_null_one_does_not() {
    let ctx = ExecutionContext::for_tests(203);
    // The outcome follows the mediator and the mediator follows the treatment.
    let strong: [(f64, f64, f64, usize); 8] = [
        (0.0, 0.0, 0.0, 72),
        (0.0, 0.0, 1.0, 8),
        (0.0, 1.0, 0.0, 2),
        (0.0, 1.0, 1.0, 18),
        (1.0, 0.0, 0.0, 18),
        (1.0, 0.0, 1.0, 2),
        (1.0, 1.0, 0.0, 8),
        (1.0, 1.0, 1.0, 72),
    ];
    // Within each arm the outcome is independent of the mediator, so the residuals of the two
    // regressions are exactly uncorrelated and the indirect effect vanishes at rho = 0.
    let null: [(f64, f64, f64, usize); 8] = [
        (0.0, 0.0, 0.0, 20),
        (0.0, 0.0, 1.0, 20),
        (0.0, 1.0, 0.0, 20),
        (0.0, 1.0, 1.0, 20),
        (1.0, 0.0, 0.0, 10),
        (1.0, 0.0, 1.0, 10),
        (1.0, 1.0, 0.0, 30),
        (1.0, 1.0, 1.0, 30),
    ];
    for (name, cells, expect_survives) in [("strong", &strong, true), ("null", &null, false)] {
        let reference = Sensitivity::of(cells);
        assert_eq!(reference.worst_and_survives().1, expect_survives, "{name}");
        let result = study_on(
            table_of(cells),
            latent_free(),
            MediationContrast::NaturalIndirect,
            InferenceMode::Frequentist,
            RefuteSuite::Cheap,
        )
        .run(&ctx)
        .unwrap();
        let sensitivity = report(&result, "mediation.sequential_ignorability");
        assert_eq!(sensitivity.passed, expect_survives, "{name}: {sensitivity:?}");
        assert!((sensitivity.comparison - reference.rho_tilde).abs() < 1e-9, "{name}");
        assert_eq!(sensitivity.failure_condition.is_some(), !expect_survives, "{name}");
    }
}

#[test]
fn the_mediator_placebo_removes_the_indirect_effect_and_keeps_the_total() {
    let ctx = ExecutionContext::for_tests(204);
    let (total, direct, indirect) =
        (counted_total(&CELLS), counted_direct(&CELLS), counted_indirect(&CELLS));
    // The reference effects are material: the placebo must have something to remove.
    assert!(indirect > 0.15 && direct.abs() > 0.02, "{direct} {indirect}");
    let result = study_on(
        table_of(&CELLS),
        latent_free(),
        MediationContrast::NaturalIndirect,
        InferenceMode::Frequentist,
        RefuteSuite::Cheap,
    )
    .run(&ctx)
    .unwrap();
    let placebo_indirect = report(&result, "mediation.placebo_mediator_indirect");
    assert!((placebo_indirect.original_ate - indirect).abs() < 1e-12);
    // A permuted mediator is independent of the treatment: nothing flows through it.
    assert!(placebo_indirect.refuted_ate.abs() < 0.07, "{placebo_indirect:?}");
    assert!(placebo_indirect.passed, "{placebo_indirect:?}");
    // ... and the direct effect is then the whole total effect.
    let placebo_direct = report(&result, "mediation.placebo_mediator_direct");
    assert!((placebo_direct.refuted_ate - total).abs() < 0.07, "{placebo_direct:?}");
    // The total effect does not read the mediator, so permuting it cannot move the total.
    let placebo_total = report(&result, "mediation.placebo_mediator_total");
    assert!((placebo_total.refuted_ate - total).abs() < 1e-12, "{placebo_total:?}");
    assert!(placebo_total.passed);
    // The published components of the subset reports are the counted decomposition.
    for (id, expected) in [
        ("mediation.subset_total", total),
        ("mediation.subset_direct", direct),
        ("mediation.subset_indirect", indirect),
    ] {
        assert!((report(&result, id).original_ate - expected).abs() < 1e-12, "{id}");
    }
}

#[test]
fn a_cpdag_whose_completions_agree_is_refuted_on_its_one_member() {
    let ctx = ExecutionContext::for_tests(205);
    let mut dag = Dag::with_variables(3);
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        dag.insert_directed(node(a), node(b)).unwrap();
    }
    let reference = Sensitivity::of(&CELLS);
    for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
        let result = study_on(
            table_of(&CELLS),
            Cpdag::from_dag(&dag),
            MediationContrast::NaturalDirect,
            InferenceMode::Frequentist,
            suite,
        )
        .run(&ctx)
        .unwrap_or_else(|e| panic!("{suite:?}: {e}"));
        // One distinct value of the requested contrast: one member, unsuffixed ids.
        assert_suite(&result, suite);
        assert!(
            (report(&result, "mediation.subset_direct").original_ate - counted_direct(&CELLS))
                .abs()
                < 1e-12
        );
        let sensitivity = report(&result, "mediation.sequential_ignorability");
        assert!((sensitivity.comparison - reference.rho_tilde).abs() < 1e-9);
        // The components still carry their own decomposition and no refutation was lost.
        let components = result.payload.mediation().expect("class decomposition");
        assert!((components.mediated.unwrap() - counted_indirect(&CELLS)).abs() < 1e-12);
    }
}

// ---------------------------------------------------------------------------
// A PAG whose candidates are all identified.
// ---------------------------------------------------------------------------

const PAG_ROWS: f64 = 20_000.0;

mod mediated_with_parents {
    use super::bern;

    pub fn p_c(c: usize) -> f64 {
        bern(0.45, c)
    }
    pub fn p_w(w: usize) -> f64 {
        bern(0.35, w)
    }
    pub fn p_t(t: usize, c: usize) -> f64 {
        bern(0.25 + 0.5 * c as f64, t)
    }
    pub fn p_m(m: usize, t: usize, w: usize) -> f64 {
        bern(0.15 + 0.4 * t as f64 + 0.3 * w as f64, m)
    }
    pub fn p_y(y: usize, t: usize, m: usize) -> f64 {
        bern(0.1 + 0.2 * t as f64 + 0.3 * m as f64 + 0.2 * (t * m) as f64, y)
    }
    pub fn p_m_given(m: usize, t: usize) -> f64 {
        (0..2).map(|w| p_w(w) * p_m(m, t, w)).sum()
    }
    pub fn joint(levels: [usize; 5]) -> f64 {
        let [c, w, t, m, y] = levels;
        p_c(c) * p_w(w) * p_t(t, c) * p_m(m, t, w) * p_y(y, t, m)
    }
    /// `E[Y(1, M(0))] - E[Y(0, M(0))]`.
    pub fn natural_direct_effect() -> f64 {
        (0..2).map(|m| p_m_given(m, 0) * (p_y(1, 1, m) - p_y(1, 0, m))).sum()
    }
}

fn mediated_data() -> TabularData {
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); 5];
    for cell in 0..32_usize {
        let mut levels = [0_usize; 5];
        for (position, level) in levels.iter_mut().enumerate() {
            *level = (cell >> (4 - position)) & 1;
        }
        let rows = (PAG_ROWS * mediated_with_parents::joint(levels)).round() as usize;
        for (column, level) in columns.iter_mut().zip(levels) {
            column.extend(std::iter::repeat_n(level as f64, rows));
        }
    }
    TabularData::from_f64_columns(
        ["c", "w", "t", "m", "y"].iter().zip(&columns).map(|(n, c)| (*n, c.as_slice())),
    )
    .unwrap()
}

/// `C -> T`; `T -> M`, `T -> Y`; `W -> M`; `M -> Y`: every mark decided, `M -> Y` visible.
fn mediated_pag() -> Pag {
    let mut pag = Pag::with_variables(5);
    for (from, to) in [(0, 2), (2, 3), (2, 4), (1, 3), (3, 4)] {
        pag.insert_directed(node(from), node(to)).unwrap();
    }
    pag
}

#[test]
fn a_pag_decomposition_is_refuted_member_by_member_against_the_structural_effect() {
    let ctx = ExecutionContext::for_tests(206);
    let truth = mediated_with_parents::natural_direct_effect();
    assert!(truth.abs() > 0.03, "the reference effect must be material: {truth}");
    let query = MediationQuery::binary(var(2), var(4), [var(3)], MediationContrast::NaturalDirect);
    for inference in [InferenceMode::Frequentist, bayesian()] {
        let built = Study::tabular(mediated_data())
            .graph(mediated_pag())
            .query(CausalQuery::Mediation(query.clone()))
            .inference(inference)
            .refute(RefuteSuite::Cheap)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        assert_eq!(built.support_status(), Some(CellStatus::Refused));
        let result = built.run(&ctx).unwrap();
        let found = ids(&result);
        // The members that identify different functionals are refuted one by one; a single
        // functional keeps the plain ids.
        let has = |prefix: &str| found.iter().any(|id| id.starts_with(prefix));
        assert!(has("mediation.sequential_ignorability"), "{found:?}");
        assert!(has("mediation.placebo_mediator_direct"), "{found:?}");
        assert!(has("mediation.subset_direct"), "{found:?}");
        assert!(has("mediation.joint_stability_subset"), "{found:?}");
        // Every member's direct effect is the structural one: a PAG whose candidates agree.
        for report in
            result.refutations.iter().filter(|r| r.refuter.starts_with("mediation.subset_direct"))
        {
            assert!((report.original_ate - truth).abs() < 6e-3, "{report:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// A temporal window.
// ---------------------------------------------------------------------------

/// The cells of the counted table as consecutive times, interleaved so neighbours differ.
fn interleaved_series() -> TimeSeriesData {
    let (mut t, mut m, mut y) = (Vec::new(), Vec::new(), Vec::new());
    let mut remaining = CELLS.to_vec();
    while remaining.iter().any(|cell| cell.3 > 0) {
        for cell in remaining.iter_mut().filter(|cell| cell.3 > 0) {
            t.push(cell.0);
            m.push(cell.1);
            y.push(cell.2);
            cell.3 -= 1;
        }
    }
    TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())],
        1,
    )
    .unwrap()
}

#[test]
fn a_temporal_admg_window_runs_the_suite_on_contiguous_stretches_of_windows() {
    let ctx = ExecutionContext::for_tests(207);
    let template = TemporalAdmg::from_static_contemporaneous(&latent_free()).unwrap();
    let reference = Sensitivity::of(&CELLS);
    for inference in [InferenceMode::Frequentist, bayesian()] {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let built = Study::series(interleaved_series())
                .graph(template.clone())
                .query(CausalQuery::Mediation(mediation_query(MediationContrast::NaturalIndirect)))
                .inference(inference.clone())
                .refute(suite)
                .bootstrap_replicates(0)
                .build()
                .unwrap();
            assert_eq!(built.support_status(), Some(CellStatus::Refused));
            let result = built.run(&ctx).unwrap_or_else(|e| panic!("{suite:?}: {e}"));
            assert_suite(&result, suite);
            let sensitivity = report(&result, "mediation.sequential_ignorability");
            assert!((sensitivity.comparison - reference.rho_tilde).abs() < 1e-9);
            // The windows are the rows of the counted table: the published components are the
            // counted decomposition.
            assert!(
                (report(&result, "mediation.subset_total").original_ate - counted_total(&CELLS))
                    .abs()
                    < 0.05
            );
            // The mediator is rotated by whole windows, not shuffled across them.
            let placebo = report(&result, "mediation.placebo_mediator_indirect");
            assert!(placebo.replicates >= 18, "{placebo:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// A TemporalPag whose indirect effect is a set of functionals.
// ---------------------------------------------------------------------------

const C: u32 = 0;
const X: u32 = 1;
const W: u32 = 2;
const M: u32 = 3;
const Y: u32 = 4;

struct Draws(u64);

impl Draws {
    fn flip(&mut self, p: f64) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        f64::from((((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64) < p)
    }
}

/// `c(t-1) -> x(t)`, `x -> m`, `w -> m`, `m -> y`, `x -> y` within a period.
fn lagged_pag_template() -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let c1 = graph.add_lagged(var(C), Lag::from_raw(1)).unwrap();
    let x0 = graph.add_lagged(var(X), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = graph.add_lagged(var(W), Lag::CONTEMPORANEOUS).unwrap();
    let m0 = graph.add_lagged(var(M), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(var(Y), Lag::CONTEMPORANEOUS).unwrap();
    for (from, to) in [(c1, x0), (x0, m0), (w0, m0), (m0, y0), (x0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

fn lagged_series(len: usize, seed: u64) -> TimeSeriesData {
    let mut draws = Draws(seed);
    let (mut c, mut x, mut w, mut m, mut y) =
        (vec![0.0; len], vec![0.0; len], vec![0.0; len], vec![0.0; len], vec![0.0; len]);
    for t in 0..len {
        c[t] = draws.flip(0.45);
        w[t] = draws.flip(0.35);
        let previous = if t == 0 { 0.0 } else { c[t - 1] };
        x[t] = draws.flip(0.25 + 0.5 * previous);
        m[t] = draws.flip(0.15 + 0.4 * x[t] + 0.3 * w[t]);
        y[t] = draws.flip(0.1 + 0.2 * x[t] + 0.3 * m[t] + 0.2 * x[t] * m[t]);
    }
    TimeSeriesData::from_f64_columns(
        [
            ("c", c.as_slice()),
            ("x", x.as_slice()),
            ("w", w.as_slice()),
            ("m", m.as_slice()),
            ("y", y.as_slice()),
        ],
        1,
    )
    .unwrap()
}

fn pag_p_m_given(m: usize, x: usize) -> f64 {
    (0..2).map(|w| bern(0.35, w) * bern(0.15 + 0.4 * x as f64 + 0.3 * w as f64, m)).sum()
}

fn pag_p_y(y: usize, x: usize, m: usize) -> f64 {
    bern(0.1 + 0.2 * x as f64 + 0.3 * m as f64 + 0.2 * (x * m) as f64, y)
}

fn structural_indirect() -> f64 {
    (0..2).map(|m| pag_p_y(1, 0, m) * (pag_p_m_given(m, 1) - pag_p_m_given(m, 0))).sum()
}

#[test]
fn a_temporal_pag_window_is_refuted_on_the_window_table_for_every_contrast() {
    let ctx = ExecutionContext::for_tests(208);
    let series = lagged_series(6_000, 23);
    let query = |contrast| MediationQuery::binary(var(X), var(Y), [var(M)], contrast);
    for contrast in [
        MediationContrast::Total,
        MediationContrast::NaturalDirect,
        MediationContrast::NaturalIndirect,
    ] {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let built = Study::series(series.clone())
                .graph(lagged_pag_template())
                .query(CausalQuery::Mediation(query(contrast)))
                .inference(InferenceMode::Frequentist)
                .refute(suite)
                .bootstrap_replicates(0)
                .build()
                .unwrap();
            assert_eq!(built.support_status(), Some(CellStatus::Refused));
            let result = built.run(&ctx).unwrap_or_else(|e| panic!("{contrast:?} {suite:?}: {e}"));
            let found = ids(&result);
            let has = |prefix: &str| found.iter().any(|id| id.starts_with(prefix));
            assert!(has("mediation.sequential_ignorability"), "{contrast:?}: {found:?}");
            assert!(has("mediation.joint_stability_subset"), "{contrast:?}: {found:?}");
            assert_eq!(has("mediation.joint_bootstrap_range"), suite == RefuteSuite::Full);
            if contrast == MediationContrast::NaturalIndirect {
                // The mediator placebo removes the indirect effect the series carries.
                let indirect = structural_indirect();
                assert!(indirect > 0.08, "{indirect}");
                for placebo in result
                    .refutations
                    .iter()
                    .filter(|r| r.refuter.starts_with("mediation.placebo_mediator_indirect"))
                {
                    assert!(placebo.refuted_ate.abs() < 0.05, "{placebo:?}");
                    assert!((placebo.original_ate - indirect).abs() < 0.05, "{placebo:?}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// A posterior over lagged temporal-ADMG templates.
// ---------------------------------------------------------------------------

const TC: usize = 0;
const TX: usize = 1;
const TW: usize = 2;
const TM: usize = 3;
const TY: usize = 4;

fn atom_posterior(weights: &[f64], atoms: &[TemporalAdmgAtom], n_vars: usize) -> GraphPosterior {
    GraphPosterior::from_atoms(
        n_vars,
        weights.to_vec(),
        atoms.iter().map(|atom| atom.contemporaneous),
        vec![0.0; n_vars * n_vars],
        vec![0.0; n_vars * n_vars],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("temporal_admg_posterior"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::TemporalAdmg)
    .with_lagged_marginals(1, vec![0.0; n_vars * n_vars])
    .unwrap()
    .with_lag_masks(atoms.iter().map(|atom| atom.lagged_directed).collect::<Vec<_>>())
    .unwrap()
    .with_lag_bidirected_masks(atoms.iter().map(|atom| atom.lagged_bidirected).collect::<Vec<_>>())
    .unwrap()
}

fn lagged(entries: &[(u32, usize, usize)], n_vars: usize) -> u64 {
    entries.iter().fold(0, |word, &(lag, from, to)| {
        with_lagged_edge(word, n_vars, lag, from, to, true).unwrap()
    })
}

fn contemporaneous(edges: &[(usize, usize)], n_vars: usize) -> AdjacencyMask {
    edges
        .iter()
        .fold(AdjacencyMask::EMPTY, |mask, &(from, to)| set_edge(mask, n_vars, from, to, true))
}

/// `c(t-1) -> x`, `x -> m`, `w -> m` within a period, `x(t-1) -> y`, `m(t-1) -> y`.
fn mediation_atom() -> TemporalAdmgAtom {
    TemporalAdmgAtom {
        contemporaneous: contemporaneous(&[(TX, TM), (TW, TM)], 5),
        lagged_directed: lagged(&[(1, TC, TX), (1, TX, TY), (1, TM, TY)], 5),
        lagged_bidirected: 0,
    }
}

/// The same template with a latent confounder of the mediator and the outcome one period later.
fn recanting_atom() -> TemporalAdmgAtom {
    TemporalAdmgAtom { lagged_bidirected: lagged(&[(1, TM, TY)], 5), ..mediation_atom() }
}

fn lagged_mediation_series(len: usize, seed: u64) -> TimeSeriesData {
    let mut draws = Draws(seed);
    let (mut c, mut x, mut w, mut m, mut y) =
        (vec![0.0; len], vec![0.0; len], vec![0.0; len], vec![0.0; len], vec![0.0; len]);
    for t in 0..len {
        c[t] = draws.flip(0.45);
        w[t] = draws.flip(0.35);
        let previous_c = if t == 0 { 0.0 } else { c[t - 1] };
        x[t] = draws.flip(0.25 + 0.5 * previous_c);
        m[t] = draws.flip(0.15 + 0.4 * x[t] + 0.3 * w[t]);
        if t > 0 {
            y[t] = draws.flip(0.1 + 0.2 * x[t - 1] + 0.3 * m[t - 1] + 0.2 * x[t - 1] * m[t - 1]);
        }
    }
    TimeSeriesData::from_f64_columns(
        [
            ("c", c.as_slice()),
            ("x", x.as_slice()),
            ("w", w.as_slice()),
            ("m", m.as_slice()),
            ("y", y.as_slice()),
        ],
        1,
    )
    .unwrap()
}

#[test]
fn a_mediation_posterior_refutes_each_atom_on_its_own_window_and_mixes_the_reports() {
    let ctx = ExecutionContext::for_tests(209);
    let series = lagged_mediation_series(20_000, 31);
    let query = MediationQuery::binary(
        var(TX as u32),
        var(TY as u32),
        [var(TM as u32)],
        MediationContrast::NaturalIndirect,
    )
    .with_horizons(vec![2])
    .unwrap();
    let posterior = || atom_posterior(&[0.6, 0.4], &[mediation_atom(), recanting_atom()], 5);
    let none = Study::series(series.clone())
        .graph_posterior(posterior())
        .query(CausalQuery::Mediation(query.clone()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!(none.refutations.is_empty());
    for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
        let built = Study::series(series.clone())
            .graph_posterior(posterior())
            .query(CausalQuery::Mediation(query.clone()))
            .refute(suite)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        assert_eq!(built.support_status(), Some(CellStatus::Refused));
        let result = built.run(&ctx).unwrap_or_else(|e| panic!("{suite:?}: {e}"));
        let found = ids(&result);
        assert!(found.iter().any(|id| id == "mediation.sequential_ignorability"), "{found:?}");
        assert!(found.iter().any(|id| id == "mediation.placebo_mediator_indirect"), "{found:?}");
        assert!(found.iter().any(|id| id == "mediation.joint_stability_subset"), "{found:?}");
        assert_eq!(
            found.iter().any(|id| id == "mediation.joint_bootstrap_range"),
            suite == RefuteSuite::Full,
            "{found:?}"
        );
        // Validation observes the mixture, it does not move it, and the recanting atom still
        // keeps its mass as unidentified mass.
        assert!((result.effect() - none.effect()).abs() < 1e-12);
        let mixture = result.structural_response.as_ref().expect("the atom mixture");
        assert!((mixture.unidentified_mass - 0.4).abs() < 1e-12);
    }
}

// ---------------------------------------------------------------------------
// A pulse contrast over the same posterior machinery.
// ---------------------------------------------------------------------------

const PZ: usize = 0;
const PX: usize = 1;
const PY: usize = 2;

fn effect_series(len: usize, seed: u64) -> TimeSeriesData {
    let mut draws = Draws(seed);
    let (mut z, mut x, mut y) = (vec![0.0; len], vec![0.0; len], vec![0.0; len]);
    for t in 0..len {
        let previous_z = if t == 0 { 0.0 } else { z[t - 1] };
        z[t] = draws.flip(0.2 + 0.6 * previous_z);
        x[t] = draws.flip(0.2 + 0.5 * z[t]);
        if t > 0 {
            y[t] = draws.flip(0.1 + 0.4 * x[t - 1] + 0.3 * previous_z);
        }
    }
    TimeSeriesData::from_f64_columns(
        [("z", z.as_slice()), ("x", x.as_slice()), ("y", y.as_slice())],
        1,
    )
    .unwrap()
}

fn true_atom() -> TemporalAdmgAtom {
    TemporalAdmgAtom {
        contemporaneous: contemporaneous(&[(PZ, PX)], 3),
        lagged_directed: lagged(&[(1, PZ, PZ), (1, PX, PY), (1, PZ, PY)], 3),
        lagged_bidirected: 0,
    }
}

fn omitting_atom() -> TemporalAdmgAtom {
    TemporalAdmgAtom {
        contemporaneous: AdjacencyMask::EMPTY,
        lagged_directed: lagged(&[(1, PZ, PZ), (1, PX, PY)], 3),
        lagged_bidirected: 0,
    }
}

#[test]
fn a_pulse_posterior_runs_the_window_suite_per_atom_and_mixes_the_reports() {
    let ctx = ExecutionContext::for_tests(210);
    let series = effect_series(4_000, 29);
    let query =
        TemporalEffectQuery::pulse(var(PX as u32), var(PY as u32), 1.0).with_horizon_steps(2);
    let posterior = || atom_posterior(&[0.7, 0.3], &[true_atom(), omitting_atom()], 3);
    let none = Study::series(series.clone())
        .graph_posterior(posterior())
        .temporal_query(query.clone())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    for (suite, expected) in [
        (RefuteSuite::Cheap, vec!["window.stretch_80", "window.placebo_treatment"]),
        (
            RefuteSuite::Full,
            vec![
                "window.stretch_80",
                "window.stretch_50",
                "window.placebo_treatment",
                "window.split_half",
            ],
        ),
    ] {
        let result = Study::series(series.clone())
            .graph_posterior(posterior())
            .temporal_query(query.clone())
            .refute(suite)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap_or_else(|e| panic!("{suite:?}: {e}"));
        let mut found = ids(&result);
        found.sort();
        let mut want: Vec<String> = expected.iter().map(|s| (*s).to_owned()).collect();
        want.sort();
        // One mixed report per refuter, not one per atom.
        assert_eq!(found, want);
        assert!((result.effect() - none.effect()).abs() < 1e-12);
        // The mixed stretch refit is the posterior-weighted mean of the two atoms' refits, which
        // are both centred on their own plug-in: it lies between the atoms' contrasts.
        let stretch = report(&result, "window.stretch_80");
        assert!(stretch.original_ate.is_finite() && stretch.refuted_ate.is_finite());
    }
}

/// A PAG decomposition publishes a contracted result that an independent consumer reads back.
#[test]
fn a_pag_decomposition_publishes_an_artifact_a_consumer_reads() {
    let ctx = ExecutionContext::for_tests(207);
    let query = MediationQuery::binary(var(2), var(4), [var(3)], MediationContrast::NaturalDirect);
    for inference in [InferenceMode::Frequentist, bayesian()] {
        let built = Study::tabular(mediated_data())
            .graph(mediated_pag())
            .query(CausalQuery::Mediation(query.clone()))
            .inference(inference)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let prepared = built.prepare(&ctx).unwrap();
        let result = prepared.estimate_retained(&ctx).unwrap();
        let artifact =
            prepared.encode_contracted_result(&result, "pag-mediation", &ctx).unwrap();
        let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
        assert_eq!(consumed.body.estimate, Some(result.effect()));
    }
}

/// The interventional distribution of the outcome on a PAG with visible edges publishes a
/// contracted result that an independent consumer reads back.
#[test]
fn a_pag_interventional_distribution_publishes_an_artifact_a_consumer_reads() {
    let ctx = ExecutionContext::for_tests(208);
    let query = antecedent_core::InterventionalDistributionQuery::new(
        var(4),
        [antecedent_core::Intervention::set(var(2), antecedent_core::Value::f64(1.0))],
    );
    let built = Study::tabular(mediated_data())
        .graph(mediated_pag())
        .query(CausalQuery::Distribution(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let prepared = built.prepare(&ctx).unwrap();
    let result = prepared.estimate_retained(&ctx).unwrap();
    assert!(result.payload.distribution().is_some());
    let artifact = prepared.encode_contracted_result(&result, "pag-distribution", &ctx).unwrap();
    let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
    assert!(consumed.body.interventional_distribution.is_some());
}

//! MediationEffect over an ADMG, the projection of a TemporalAdmg and a CPDAG.
//!
//! The expected values are counted from one contingency table with the mediation formulas of
//! a latent-free graph (Pearl 2001), not read from the engine:
//! `NDE = sum_m [E(Y|1,m) - E(Y|0,m)] P(m|0)`,
//! `NIE = sum_m E(Y|0,m) [P(m|1) - P(m|0)]` (the pure natural indirect effect) and
//! `TE = E(Y|1) - E(Y|0)`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, InterventionalDistributionQuery,
    MediationContrast, MediationQuery, Value, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, TemporalAdmg};

/// `(t, m, y, count)` of the table.
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

fn table() -> TabularData {
    let (mut t, mut m, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for (tv, mv, yv, count) in CELLS {
        for _ in 0..count {
            t.push(tv);
            m.push(mv);
            y.push(yv);
        }
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

fn count(t: Option<f64>, m: Option<f64>, y: Option<f64>) -> f64 {
    CELLS
        .iter()
        .filter(|(tv, mv, yv, _)| {
            t.is_none_or(|v| v == *tv) && m.is_none_or(|v| v == *mv) && y.is_none_or(|v| v == *yv)
        })
        .map(|(_, _, _, n)| *n as f64)
        .sum()
}

fn e_y(t: f64, m: f64) -> f64 {
    count(Some(t), Some(m), Some(1.0)) / count(Some(t), Some(m), None)
}

fn p_m(m: f64, t: f64) -> f64 {
    count(Some(t), Some(m), None) / count(Some(t), None, None)
}

fn counted_total() -> f64 {
    count(Some(1.0), None, Some(1.0)) / count(Some(1.0), None, None)
        - count(Some(0.0), None, Some(1.0)) / count(Some(0.0), None, None)
}

fn counted_direct() -> f64 {
    [0.0, 1.0].into_iter().map(|m| (e_y(1.0, m) - e_y(0.0, m)) * p_m(m, 0.0)).sum()
}

fn counted_indirect() -> f64 {
    [0.0, 1.0].into_iter().map(|m| e_y(0.0, m) * (p_m(m, 1.0) - p_m(m, 0.0))).sum()
}

fn query(contrast: MediationContrast) -> MediationQuery {
    MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        contrast,
    )
}

fn latent_free() -> Admg {
    let mut admg = Admg::with_variables(3);
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        admg.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    admg
}

fn study(
    graph: impl antecedent::IntoGraphInput,
    contrast: MediationContrast,
    inference: InferenceMode,
    replicates: u32,
) -> Study {
    Study::tabular(table())
        .graph(graph)
        .query(CausalQuery::Mediation(query(contrast)))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap()
}

fn joint(result: &antecedent::StudyResult) -> Option<&antecedent_core::Diagnostic> {
    result.diagnostics.iter().find(|d| d.code.as_ref() == "estimate.mediation.joint")
}

fn field(diagnostic: &antecedent_core::Diagnostic, key: &str) -> f64 {
    diagnostic
        .fields
        .iter()
        .find(|(k, _)| k.as_ref() == key)
        .unwrap_or_else(|| panic!("field {key}"))
        .1
        .parse()
        .unwrap()
}

#[test]
fn an_admg_decomposition_is_the_counted_total_direct_and_indirect_effect() {
    let ctx = ExecutionContext::for_tests(61);
    let (total, direct, indirect) = (counted_total(), counted_direct(), counted_indirect());
    // 33/65 - 25/60.
    assert!((total - 0.091_026).abs() < 1e-5, "{total}");
    for (contrast, expected) in [
        (MediationContrast::Total, total),
        (MediationContrast::NaturalDirect, direct),
        (MediationContrast::NaturalIndirect, indirect),
    ] {
        let built = study(latent_free(), contrast, InferenceMode::Frequentist, 60);
        assert_eq!(built.support_status(), Some(CellStatus::Refused));
        let result = built.run(&ctx).unwrap_or_else(|e| panic!("{contrast:?}: {e}"));
        assert!((result.effect() - expected).abs() < 1e-12, "{contrast:?}: {}", result.effect());
        let components = result.payload().mediation().expect("the decomposition");
        assert!((components.total.unwrap() - total).abs() < 1e-12);
        assert!((components.direct.unwrap() - direct).abs() < 1e-12);
        assert!((components.mediated.unwrap() - indirect).abs() < 1e-12);
        // The components were bootstrapped on the same resampled tables.
        let diagnostic = joint(&result).expect("joint disclosure");
        assert!(field(diagnostic, "draws") >= 40.0);
        assert!(field(diagnostic, "total_sd") > 0.0);
        assert!(field(diagnostic, "direct_sd") > 0.0);
        let correlation = field(diagnostic, "corr_direct_indirect");
        assert!((-1.0..=1.0).contains(&correlation), "{correlation}");
        // The requested contrast carries its own standard error.
        let se = result.estimate().as_effect().unwrap().se_bootstrap().expect("a bootstrap SE");
        let reported = field(
            diagnostic,
            match contrast {
                MediationContrast::Total => "total_sd",
                MediationContrast::NaturalDirect => "direct_sd",
                _ => "indirect_sd",
            },
        );
        assert!((se - reported).abs() <= 0.05 * reported + 1e-9, "{se} vs {reported}");
    }
}

#[test]
fn the_bayesian_bootstrap_pairs_the_component_posteriors() {
    let ctx = ExecutionContext::for_tests(62);
    let inference = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(400));
    let result =
        study(latent_free(), MediationContrast::NaturalDirect, inference, 0).run(&ctx).unwrap();
    let posterior = result.posterior.as_ref().expect("a posterior");
    assert_eq!(posterior.draws.n_draws, 400);
    assert!((result.effect() - counted_direct()).abs() < 0.05, "{}", result.effect());
    let diagnostic = joint(&result).expect("joint disclosure");
    assert_eq!(field(diagnostic, "draws"), 400.0);
    assert!((field(diagnostic, "total_mean") - counted_total()).abs() < 0.05);
    assert!((field(diagnostic, "indirect_mean") - counted_indirect()).abs() < 0.05);
    assert!(field(diagnostic, "direct_sd") > 0.0);
}

#[test]
fn the_contemporaneous_projection_and_a_fully_oriented_cpdag_give_the_same_decomposition() {
    let ctx = ExecutionContext::for_tests(63);
    let temporal = TemporalAdmg::from_static_contemporaneous(&latent_free()).unwrap();
    let mut dag = Dag::with_variables(3);
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let cpdag = Cpdag::from_dag(&dag);
    for contrast in [
        MediationContrast::Total,
        MediationContrast::NaturalDirect,
        MediationContrast::NaturalIndirect,
    ] {
        let expected = match contrast {
            MediationContrast::Total => counted_total(),
            MediationContrast::NaturalDirect => counted_direct(),
            _ => counted_indirect(),
        };
        let projected = study(temporal.clone(), contrast, InferenceMode::Frequentist, 0)
            .run(&ctx)
            .unwrap_or_else(|e| panic!("temporal {contrast:?}: {e}"));
        assert!((projected.effect() - expected).abs() < 1e-12, "{contrast:?}");
        let class = study(cpdag.clone(), contrast, InferenceMode::Frequentist, 0)
            .run(&ctx)
            .unwrap_or_else(|e| panic!("cpdag {contrast:?}: {e}"));
        assert!((class.effect() - expected).abs() < 1e-12, "{contrast:?}: {}", class.effect());
        let components = class.payload().mediation().expect("class decomposition");
        assert!((components.total.unwrap() - counted_total()).abs() < 1e-12);
        assert!((components.direct.unwrap() - counted_direct()).abs() < 1e-12);
    }
}

#[test]
fn a_confounded_treatment_outcome_pair_leaves_the_requested_effect_not_identified() {
    // t -> m -> y and t -> y with t <-> y: the total effect has no adjustment, no instrument
    // and no front door (the direct edge blocks it).
    let mut admg = latent_free();
    admg.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    let ctx = ExecutionContext::for_tests(64);
    let refusal = study(admg, MediationContrast::Total, InferenceMode::Frequentist, 0)
        .run(&ctx)
        .unwrap_err()
        .to_string();
    assert!(refusal.contains("effect_not_identified"), "{refusal}");
}

#[test]
fn the_front_door_total_effect_is_identified_while_the_direct_component_is_absent() {
    // t -> m -> y with t <-> y and no direct edge: the front-door formula identifies the total
    // effect; the natural direct effect is nil and no longer part of the question.
    let mut admg = Admg::with_variables(3);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    admg.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    admg.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    // Front door: sum_m P(m|t) sum_t' P(y=1|m,t') P(t').
    let p_t = |t: f64| count(Some(t), None, None) / count(None, None, None);
    let inner = |m: f64| -> f64 { [0.0, 1.0].into_iter().map(|t| e_y(t, m) * p_t(t)).sum() };
    let front = |t: f64| -> f64 { [0.0, 1.0].into_iter().map(|m| p_m(m, t) * inner(m)).sum() };
    let ctx = ExecutionContext::for_tests(65);
    let result =
        study(admg, MediationContrast::Total, InferenceMode::Frequentist, 0).run(&ctx).unwrap();
    assert!((result.effect() - (front(1.0) - front(0.0))).abs() < 1e-12, "{}", result.effect());
}

/// One period, no lagged cause: the window of the pulse is a single time point, so the series
/// of the contingency table read as consecutive times has the counted decomposition. The
/// decomposition is estimated on windows, and a latent cause from before the window is part
/// of the template's identification.
#[test]
fn a_temporal_admg_window_with_one_period_reproduces_the_counted_decomposition() {
    let (mut t, mut m, mut y) = (Vec::new(), Vec::new(), Vec::new());
    // Interleave the cells so consecutive times differ.
    let mut remaining: Vec<(f64, f64, f64, usize)> = CELLS.to_vec();
    while remaining.iter().any(|cell| cell.3 > 0) {
        for cell in remaining.iter_mut().filter(|cell| cell.3 > 0) {
            t.push(cell.0);
            m.push(cell.1);
            y.push(cell.2);
            cell.3 -= 1;
        }
    }
    let series = TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())],
        1,
    )
    .unwrap();
    let template = TemporalAdmg::from_static_contemporaneous(&latent_free()).unwrap();
    let ctx = ExecutionContext::for_tests(66);
    for contrast in [
        MediationContrast::Total,
        MediationContrast::NaturalDirect,
        MediationContrast::NaturalIndirect,
    ] {
        let expected = match contrast {
            MediationContrast::Total => counted_total(),
            MediationContrast::NaturalDirect => counted_direct(),
            _ => counted_indirect(),
        };
        let built = Study::series(series.clone())
            .graph(template.clone())
            .query(CausalQuery::Mediation(query(contrast)))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(40)
            .build()
            .unwrap_or_else(|e| panic!("{contrast:?}: {e}"));
        let result = built.run(&ctx).unwrap_or_else(|e| panic!("{contrast:?}: {e}"));
        assert!((result.effect() - expected).abs() < 1e-12, "{contrast:?}: {}", result.effect());
        let components = result.payload().mediation().expect("the decomposition");
        assert!((components.total.unwrap() - counted_total()).abs() < 1e-12);
        assert!((components.direct.unwrap() - counted_direct()).abs() < 1e-12);
        assert!((components.mediated.unwrap() - counted_indirect()).abs() < 1e-12);
        assert!(joint(&result).is_some());
        assert!(
            result.diagnostics.iter().any(|d| d.code.as_ref() == "estimate.temporal_admg.window")
        );
    }
}

fn distribution_study(graph: impl antecedent::IntoGraphInput) -> Study {
    Study::tabular(table())
        .graph(graph)
        .query(CausalQuery::Distribution(InterventionalDistributionQuery::new(
            VariableId::from_raw(2),
            [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
        )))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(30)
        .build()
        .unwrap()
}

fn probability_of_one(result: &antecedent::StudyResult) -> f64 {
    result
        .payload()
        .distribution()
        .expect("a distribution")
        .atoms
        .iter()
        .find(|atom| atom.outcomes[0].1 == Value::f64(1.0))
        .expect("the y = 1 atom")
        .probability
}

#[test]
fn a_cpdag_whose_completions_agree_gives_the_counted_interventional_distribution() {
    let mut dag = Dag::with_variables(3);
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let ctx = ExecutionContext::for_tests(67);
    let built = distribution_study(Cpdag::from_dag(&dag));
    assert_eq!(built.support_status(), Some(CellStatus::Refused));
    let result = built.run(&ctx).unwrap();
    // P(y = 1 | do(t = 1)) = sum_m P(m | t = 1) P(y = 1 | t = 1, m).
    let expected: f64 = [0.0, 1.0].into_iter().map(|m| p_m(m, 1.0) * e_y(1.0, m)).sum();
    assert!((probability_of_one(&result) - expected).abs() < 1e-12);
    assert!(
        result
            .diagnostics
            .iter()
            .all(|d| d.code.as_ref() != "estimate.distribution.class_envelope")
    );
}

#[test]
fn completions_that_disagree_publish_the_identified_set_of_each_atom() {
    // t -- y with m isolated: the completions are t -> y (P(y | t)) and y -> t (P(y)).
    let mut cpdag = Cpdag::with_variables(3);
    cpdag.insert_undirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    let conditional = count(Some(1.0), None, Some(1.0)) / count(Some(1.0), None, None);
    let marginal = count(None, None, Some(1.0)) / count(None, None, None);
    let ctx = ExecutionContext::for_tests(68);
    let result = distribution_study(cpdag).run(&ctx).unwrap();
    // Equal enumeration weights: the published atom is the mean of the two members.
    assert!(
        (probability_of_one(&result) - 0.5 * (conditional + marginal)).abs() < 1e-12,
        "{}",
        probability_of_one(&result)
    );
    let envelope = result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "estimate.distribution.class_envelope")
        .expect("the per-atom envelope");
    let value = |key: &str| -> Option<f64> {
        envelope.fields.iter().find(|(k, _)| k.as_ref() == key).map(|(_, v)| v.parse().unwrap())
    };
    let (low, high) = (conditional.min(marginal), conditional.max(marginal));
    let found = (0..2).any(|index| {
        value(&format!("atom_{index}_lower")).is_some_and(|l| (l - low).abs() < 1e-12)
            && value(&format!("atom_{index}_upper")).is_some_and(|u| (u - high).abs() < 1e-12)
    });
    assert!(found, "{:?}", envelope.fields);
}

/// Each mixed-graph coordinate publishes a contracted result that an independent consumer reads
/// back: the decomposition on an ADMG, a CPDAG and a temporal ADMG, and the
/// interventional distribution on an ADMG and a CPDAG.
#[test]
fn mixed_graph_mediation_and_distribution_publish_an_artifact_a_consumer_reads() {
    let ctx = ExecutionContext::for_tests(69);
    let mut dag = Dag::with_variables(3);
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let mut interleaved = (Vec::new(), Vec::new(), Vec::new());
    let mut remaining: Vec<(f64, f64, f64, usize)> = CELLS.to_vec();
    while remaining.iter().any(|cell| cell.3 > 0) {
        for cell in remaining.iter_mut().filter(|cell| cell.3 > 0) {
            interleaved.0.push(cell.0);
            interleaved.1.push(cell.1);
            interleaved.2.push(cell.2);
            cell.3 -= 1;
        }
    }
    let series = TimeSeriesData::from_f64_columns(
        [
            ("t", interleaved.0.as_slice()),
            ("m", interleaved.1.as_slice()),
            ("y", interleaved.2.as_slice()),
        ],
        1,
    )
    .unwrap();
    let template = TemporalAdmg::from_static_contemporaneous(&latent_free()).unwrap();
    let mediation = |study: antecedent::StudyBuilder| {
        study
            .query(CausalQuery::Mediation(query(MediationContrast::NaturalDirect)))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(20)
            .build()
            .unwrap()
    };
    let cases = [
        ("admg-mediation", mediation(Study::tabular(table()).graph(latent_free()))),
        ("cpdag-mediation", mediation(Study::tabular(table()).graph(Cpdag::from_dag(&dag)))),
        ("temporal-admg-mediation", mediation(Study::series(series).graph(template))),
        ("admg-distribution", distribution_study(latent_free())),
        ("dag-distribution", distribution_study(dag.clone())),
        ("cpdag-distribution", distribution_study(Cpdag::from_dag(&dag))),
    ];
    for (label, built) in cases {
        let prepared = built.prepare(&ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
        let result = prepared.estimate_retained(&ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
        let artifact = prepared
            .encode_contracted_result(&result, &format!("mixed-graph-{label}"), &ctx)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let consumed = antecedent_io::consume_analysis_result(&artifact)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        if label.ends_with("mediation") {
            assert_eq!(consumed.body.estimate, Some(result.effect()), "{label}");
        }
    }
}

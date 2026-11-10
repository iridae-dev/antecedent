//! Implied-independence staleness diagnostic: a hidden common cause that moves a measured
//! column is found and asks to identify again, the false-alarm rate over unchanged samples
//! stays at the declared level, a graph that implies nothing is reported as untested, and the
//! exported baseline gives the same report without the producer.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::recalc::Stage;
use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_data::TabularData;
use antecedent_kernels::standard_normal;

use super::{GraphImplications, ImpliedIndependence};
use crate::staleness::{
    FindingKind, Recommendation, StalenessBaseline, StalenessPlan, StalenessPolicy,
    StalenessReport, assess_exported_baseline,
};

const LABEL: &str = "implied_independence";

/// Customers of a subscription business. `price -> engagement <- tenure`, `price -> support`,
/// and `cancellation` depends on engagement, support and price. With `hidden`, an unmeasured
/// cause raises engagement, support and cancellation together.
fn customers(n: usize, seed: u64, hidden: bool) -> Vec<(&'static str, Vec<f64>)> {
    let mut rng = CausalRng::from_seed(seed);
    let (mut price, mut tenure, mut engagement, mut support, mut cancellation) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let p = standard_normal(&mut rng);
        let t = standard_normal(&mut rng);
        let u = if hidden { standard_normal(&mut rng) } else { 0.0 };
        let e = -0.6 * p + 0.5 * t + 0.6 * u + standard_normal(&mut rng);
        let s = 0.5 * p + 0.8 * u + standard_normal(&mut rng);
        let c = -0.8 * e + 0.5 * s + 0.3 * p + 0.8 * u + standard_normal(&mut rng);
        price.push(p);
        tenure.push(t);
        engagement.push(e);
        support.push(s);
        cancellation.push(c);
    }
    vec![
        ("price", price),
        ("tenure", tenure),
        ("engagement", engagement),
        ("support", support),
        ("cancellation", cancellation),
    ]
}

fn table(columns: &[(&'static str, Vec<f64>)]) -> TabularData {
    TabularData::from_f64_columns(columns.iter().map(|(n, v)| (*n, v.as_slice()))).unwrap()
}

fn only(columns: &[(&'static str, Vec<f64>)], keep: &[&str]) -> TabularData {
    let kept: Vec<_> = columns.iter().filter(|(n, _)| keep.contains(n)).cloned().collect();
    table(&kept)
}

fn statement(x: &str, y: &str, given: &[&str]) -> ImpliedIndependence {
    ImpliedIndependence {
        x: x.into(),
        y: y.into(),
        given: given.iter().map(|g| (*g).to_owned()).collect(),
    }
}

/// What the declared graph implies: one statement per pair it separates.
fn graph_statements() -> Vec<ImpliedIndependence> {
    vec![
        statement("price", "tenure", &[]),
        statement("tenure", "support", &[]),
        statement("tenure", "cancellation", &["price", "engagement"]),
        statement("engagement", "support", &["price"]),
    ]
}

fn plan(implications: Option<GraphImplications>) -> StalenessPlan {
    StalenessPlan {
        plan_identity: "plan:engagement->cancellation|price".into(),
        treatment: Some("engagement".into()),
        outcome: Some("cancellation".into()),
        adjustment: vec!["price".into()],
        bayesian: false,
        implications,
    }
}

fn graph_plan() -> StalenessPlan {
    plan(Some(GraphImplications::Listed(graph_statements())))
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(7)
}

fn assess(
    plan: &StalenessPlan,
    old: &TabularData,
    new: &TabularData,
    policy: &StalenessPolicy,
) -> StalenessReport {
    StalenessBaseline::build(old, plan, policy, &ctx())
        .unwrap()
        .assess(new, policy, &ctx())
        .unwrap()
}

fn skipped<'a>(report: &'a StalenessReport, what: &str) -> Option<&'a str> {
    report.not_evaluated.iter().find(|(name, _)| name == what).map(|(_, why)| why.as_str())
}

#[test]
fn a_hidden_cause_that_moves_a_measured_column_asks_to_reidentify() {
    let policy = StalenessPolicy::default();
    let january = table(&customers(4000, 1, false));
    let april = table(&customers(4000, 2, true));
    let report = assess(&graph_plan(), &january, &april, &policy);
    assert_eq!(report.recommendation, Recommendation::Reidentify, "{report:#?}");
    assert!(report.changed_keys.contains(&Stage::Evidence));
    assert!(report.next_attempt.contains("review the graph"), "{}", report.next_attempt);

    let failed = report
        .findings()
        .find(|f| f.subject == "engagement _||_ support | price")
        .expect("the independence the hidden cause breaks is a finding");
    assert_eq!(failed.kind, FindingKind::ImpliedIndependence);
    assert_eq!(failed.forces, Recommendation::Reidentify);
    assert_eq!(failed.threshold_name, "implication_threshold");
    // Given price, engagement keeps 0.5 tenure + 0.6 u + noise and support keeps 0.8 u + noise:
    // covariance 0.48, variances 1.61 and 1.64, so the partial correlation is 0.2954.
    let expected = 0.48 / (1.61_f64 * 1.64).sqrt();
    assert!((failed.statistic - expected).abs() < 0.05, "{} vs {expected}", failed.statistic);
    let recorded = failed.auxiliary.iter().find(|(name, _)| name == "recorded_statistic").unwrap();
    assert!(recorded.1.abs() < 0.05, "the independence held on the plan's data: {}", recorded.1);

    // The statements the hidden cause leaves alone are evaluated and do not exceed.
    for subject in ["price _||_ tenure", "tenure _||_ support"] {
        let kept = report.evaluations.iter().find(|f| f.subject == subject).unwrap();
        assert!(!kept.exceeded, "{subject}: {kept:?}");
    }
    // No other diagnostic reads the graph: without its implications nothing asks to identify
    // again, whatever the marginal distributions did.
    let without = assess(&plan(None), &january, &april, &policy);
    assert!(without.recommendation < Recommendation::Reidentify, "{without:#?}");
    assert!(without.findings().all(|f| f.kind != FindingKind::ImpliedIndependence));
    assert_eq!(skipped(&without, LABEL), Some("the baseline records no graph for the plan"));
}

#[test]
fn an_unchanged_mechanism_is_reused_and_false_alarms_stay_at_the_declared_level() {
    let policy = StalenessPolicy { alpha: 0.05, ..StalenessPolicy::default() };
    let seeds = 100u64;
    let mut alarms = 0u64;
    for seed in 0..seeds {
        let report = assess(
            &graph_plan(),
            &table(&customers(1000, 1000 + seed, false)),
            &table(&customers(1000, 5000 + seed, false)),
            &policy,
        );
        let implied =
            report.evaluations.iter().filter(|f| f.kind == FindingKind::ImpliedIndependence);
        assert!(implied.count() >= 3, "the statements are evaluated on every seed");
        if report.findings().any(|f| f.kind == FindingKind::ImpliedIndependence) {
            alarms += 1;
        }
    }
    // alpha plus three binomial standard errors.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the binomial bound converts a small count directly"
    )]
    let bound = (seeds as f64 * 0.05 + 3.0 * (seeds as f64 * 0.05 * 0.95).sqrt()).floor() as u64;
    assert!(alarms <= bound, "false alarms {alarms} of {seeds} exceed {bound}");
}

#[test]
fn a_graph_that_implies_nothing_is_reported_as_untested_not_as_passed() {
    // price -> engagement -> cancellation with price -> cancellation joins every pair.
    let keep = ["price", "engagement", "cancellation"];
    let january = only(&customers(4000, 1, false), &keep);
    let april = only(&customers(4000, 2, true), &keep);
    let report = assess(
        &plan(Some(GraphImplications::Listed(Vec::new()))),
        &january,
        &april,
        &StalenessPolicy::default(),
    );
    assert_eq!(report.recommendation, Recommendation::Reuse, "{report:#?}");
    let reason = skipped(&report, LABEL).expect("the untested diagnostic is named");
    assert!(reason.contains("implies no conditional independence"), "{reason}");
    assert!(report.evaluations.iter().all(|f| f.kind != FindingKind::ImpliedIndependence));
}

#[test]
fn a_statement_the_plan_data_already_contradicted_is_not_a_staleness_finding() {
    let policy = StalenessPolicy::default();
    let report = assess(
        &graph_plan(),
        &table(&customers(4000, 2, true)),
        &table(&customers(4000, 3, true)),
        &policy,
    );
    let reason = skipped(&report, "engagement _||_ support | price").unwrap();
    assert!(reason.contains("already contradicted"), "{reason}");
    assert!(report.findings().all(|f| f.subject != "engagement _||_ support | price"));
}

#[test]
fn unavailable_implications_a_cap_and_a_different_test_are_named_with_their_reason() {
    let january = table(&customers(600, 1, false));
    let april = table(&customers(600, 2, true));
    let policy = StalenessPolicy::default();

    let unavailable = plan(Some(GraphImplications::Unavailable("the graph is temporal".into())));
    let report = assess(&unavailable, &january, &april, &policy);
    assert_eq!(skipped(&report, LABEL), Some("the graph is temporal"));

    let capped = StalenessPolicy { max_implications: 2, ..StalenessPolicy::default() };
    let report = assess(&graph_plan(), &january, &april, &capped);
    assert!(skipped(&report, LABEL).unwrap().contains("more than max_implications (2)"));

    let baseline = StalenessBaseline::build(&january, &graph_plan(), &policy, &ctx()).unwrap();
    let other =
        StalenessPolicy { implication_test: "robust_parcorr".into(), ..StalenessPolicy::default() };
    let report = baseline.assess(&april, &other, &ctx()).unwrap();
    assert!(skipped(&report, LABEL).unwrap().contains("recorded the parcorr test"));

    // A statement over a column the new data lacks cannot be tested, and says so; the schema
    // change is reported by its own diagnostic.
    let fewer = only(&customers(600, 2, true), &["price", "engagement", "support", "cancellation"]);
    let report = baseline.assess(&fewer, &policy, &ctx()).unwrap();
    let reason = skipped(&report, "price _||_ tenure").unwrap();
    assert!(reason.contains("cannot be tested on the new data"), "{reason}");
}

#[test]
fn a_baseline_without_a_graph_keeps_its_features_and_one_with_a_graph_declares_them() {
    let january = table(&customers(400, 1, false));
    let policy = StalenessPolicy::default();
    let plain = StalenessBaseline::build(&january, &plan(None), &policy, &ctx()).unwrap();
    assert!(plain.wire().implications.is_none());
    assert_eq!(plain.wire().required_features, ["staleness_baseline_v1"]);
    let graph = StalenessBaseline::build(&january, &graph_plan(), &policy, &ctx()).unwrap();
    assert_eq!(
        graph.wire().required_features,
        ["staleness_baseline_v1", "staleness_graph_implications_v1"]
    );
    let recorded = graph.wire().implications.as_ref().unwrap();
    assert_eq!(recorded.test, "parcorr");
    assert_eq!(recorded.items.len(), 4);
    assert!(recorded.items.iter().all(|item| item.statistic.is_some() && item.p_value.is_some()));
    assert_ne!(plain.identity(), graph.identity());
}

#[test]
fn an_exported_baseline_and_any_thread_count_give_the_same_report() {
    let policy = StalenessPolicy::default();
    let january = table(&customers(2000, 1, false));
    let april = table(&customers(2000, 2, true));
    let baseline = StalenessBaseline::build(&january, &graph_plan(), &policy, &ctx()).unwrap();
    let reference = baseline.assess(&april, &policy, &ExecutionContext::for_tests(1)).unwrap();
    assert_eq!(reference.recommendation, Recommendation::Reidentify);
    let bytes = baseline.export().unwrap();
    let consumed = assess_exported_baseline(&bytes, &april, &policy, &ctx()).unwrap();
    assert_eq!(consumed.identity, reference.identity);
    for threads in [1u32, 2, 8] {
        let report =
            baseline.assess(&april, &policy, &ExecutionContext::production(99, threads)).unwrap();
        assert_eq!(report.identity, reference.identity);
    }
}

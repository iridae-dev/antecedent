//! Known-truth evidence for licensed static cells, one table-driven test per
//! family.
//!
//! Each test walks its family's structure × validation (and, where both are
//! licensed, inference) coordinates through the staged `Study` handle
//! (`prepare` → `estimate`), compares the licensed number with a frozen
//! conformance fixture, and checks that the requested validation suite actually
//! ran (`none` runs nothing; `cheap` / `full` attach the reports the cell's
//! limitations name).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

use std::sync::Arc;

use antecedent::validate::PredictiveCheckKind;
use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, CounterfactualQuery, ExecutionContext,
    IdentificationStatus, Intervention, InterventionalDistributionQuery, MediationContrast,
    MediationQuery, Value, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};

mod common;

// The three-atom static graph posterior, in one owner.
use common::fixtures::mixture_graph_posterior;

const SUITES: [RefuteSuite; 3] = [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full];

fn json(text: &str) -> serde_json::Value {
    serde_json::from_str(text).unwrap()
}

fn dag(n: u32, edges: &[(u32, u32)]) -> Dag {
    let mut dag = Dag::with_variables(n);
    for &(s, t) in edges {
        dag.insert_directed(DenseNodeId::from_raw(s), DenseNodeId::from_raw(t)).unwrap();
    }
    dag
}

fn bayes(n_draws: usize, prior_scale: f64) -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(n_draws).prior_scale(prior_scale))
}

/// Staged run on an explicit or accepted DAG: `prepare` then one `estimate`
/// click, which must reuse the frozen identification.
fn staged(
    data: &TabularData,
    graph: &Dag,
    accepted: bool,
    query: impl Into<CausalQuery>,
    inference: InferenceMode,
    suite: RefuteSuite,
    seed: u64,
) -> StudyResult {
    let builder = Study::tabular(data.clone());
    let builder = if accepted {
        builder.graph(AcceptedGraph::from(graph.clone()))
    } else {
        builder.graph(graph.clone())
    };
    let study = builder
        .query(query)
        .inference(inference)
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    assert_eq!(study.structure_source().as_str(), if accepted { "accepted" } else { "explicit" });
    let ctx = ExecutionContext::for_tests(seed);
    let result = study.prepare(&ctx).unwrap().estimate(data, &ctx).unwrap();
    assert_eq!(result.support_status.unwrap().as_str(), "licensed");
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
        "the prepared click must reuse identification"
    );
    result
}

fn refuter_names(result: &StudyResult) -> Vec<&str> {
    result.refutations.iter().map(|r| r.refuter.as_ref()).collect()
}

fn report<'a>(result: &'a StudyResult, name: &str) -> &'a antecedent_validate::RefutationReport {
    result
        .refutations
        .iter()
        .find(|r| r.refuter.as_ref() == name)
        .unwrap_or_else(|| panic!("{name} must run; got {:?}", refuter_names(result)))
}

/// The typed E-value pair must reproduce the `sensitivity.evalue` verdict without
/// reading report prose: `evalue` mirrors the report's own number and
/// `evalue_threshold` is the value that report judged it against. Both stay `None`
/// when the refuter did not run.
fn assert_evalue_fields(result: &StudyResult, ran: bool, label: &str) {
    if !ran {
        assert_eq!(result.estimate.evalue, None, "{label}");
        assert_eq!(result.estimate.evalue_threshold, None, "{label}");
        return;
    }
    let report = report(result, "sensitivity.evalue");
    assert_eq!(result.estimate.evalue, Some(report.comparison), "{label}");
    let threshold = result.estimate.evalue_threshold.expect("threshold must accompany the evalue");
    assert_eq!(threshold, antecedent_validate::DEFAULT_EVALUE_THRESHOLD, "{label}");
    assert_eq!(
        report.passed,
        report.comparison >= threshold,
        "{label}: the typed pair must reproduce the report verdict"
    );
}

/// Bayesian cheap/full attach prior and posterior predictive checks; full also
/// runs prior sensitivity. `none` runs no report of any kind.
fn assert_bayesian_validation(result: &StudyResult, suite: RefuteSuite, label: &str) {
    let has = |kind| result.predictive_checks.iter().any(|c| c.kind == kind);
    let sensitivity = result.posterior.as_ref().and_then(|p| p.prior_sensitivity.as_ref());
    if suite == RefuteSuite::None {
        assert!(result.refutations.is_empty(), "{label}: none runs no refuter");
        assert!(result.predictive_checks.is_empty(), "{label}: none runs no PPC");
        assert!(sensitivity.is_none(), "{label}: none runs no prior sensitivity");
    } else {
        assert!(!result.refutations.is_empty(), "{label}: refuters must run");
        assert!(has(PredictiveCheckKind::Prior), "{label}: prior PPC must run");
        assert!(has(PredictiveCheckKind::Posterior), "{label}: posterior PPC must run");
        assert_eq!(
            sensitivity.is_some(),
            suite == RefuteSuite::Full,
            "{label}: prior sensitivity runs under full only"
        );
    }
}

// ---------------------------------------------------------------- AverageEffect

/// `conformance/estimate/linear_gaussian_ate`: `y = 1 + 2t + 3z`,
/// `t = 1{z > 0.5}`, `z = i/n`, so the adjusted ATE is exactly 2.
fn linear_gaussian_ate() -> (TabularData, Dag, AverageEffectQuery, serde_json::Value) {
    let expected =
        json(include_str!("../../../conformance/estimate/linear_gaussian_ate/expected.json"));
    let csv = include_str!("../../../conformance/estimate/linear_gaussian_ate/data.csv");
    let mut columns: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for line in csv.lines().skip(1) {
        for (column, cell) in columns.iter_mut().zip(line.split(',')) {
            column.push(cell.parse().unwrap());
        }
    }
    assert_eq!(columns[0].len() as u64, expected["n"].as_u64().unwrap());
    let data = TabularData::from_f64_columns([
        ("t", columns[0].as_slice()),
        ("y", columns[1].as_slice()),
        ("z", columns[2].as_slice()),
    ])
    .unwrap();
    let graph = dag(3, &[(2, 0), (2, 1), (0, 1)]);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (data, graph, query, expected)
}

/// Frequentist `AverageEffect × Dag` on explicit and accepted structure under
/// every validation level. The linear-adjustment ATE must equal the fixture's
/// analytic truth; cheap must run overlap and the E-value; full must also run
/// the placebo and random-common-cause refuters inside the bands pinned by
/// `conformance/validate/refuters` (same SCM).
#[test]
fn average_effect_dag_frequentist_known_truth_all_structures_and_suites() {
    let (data, graph, query, expected) = linear_gaussian_ate();
    let truth = expected["true_ate"].as_f64().unwrap();
    let refuters = json(include_str!("../../../conformance/validate/refuters/expected.json"));
    let bands = &refuters["expected"];
    assert_eq!(bands["original_ate"].as_f64().unwrap(), truth);
    for accepted in [false, true] {
        for suite in SUITES {
            let label = format!("accepted={accepted} suite={suite:?}");
            let result = staged(
                &data,
                &graph,
                accepted,
                query.clone(),
                InferenceMode::Frequentist,
                suite,
                7,
            );
            assert!(result.posterior.is_none(), "{label}: Frequentist publishes no posterior");
            assert!((result.estimate.ate - truth).abs() < 1e-8, "{label}: {}", result.estimate.ate);
            assert_evalue_fields(&result, suite != RefuteSuite::None, &label);
            match suite {
                RefuteSuite::None => assert!(result.refutations.is_empty(), "{label}"),
                RefuteSuite::Cheap => {
                    let mut names = refuter_names(&result);
                    names.sort_unstable();
                    assert_eq!(names, ["overlap.assessment", "sensitivity.evalue"], "{label}");
                }
                _ => {
                    report(&result, "overlap.assessment");
                    report(&result, "sensitivity.evalue");
                    let placebo = report(&result, "placebo.treatment");
                    assert!(
                        placebo.refuted_ate.abs() < bands["placebo_abs_max"].as_f64().unwrap(),
                        "{label}: placebo effect {}",
                        placebo.refuted_ate
                    );
                    let rcc = report(&result, "random.common_cause");
                    assert!(
                        (rcc.refuted_ate - rcc.original_ate).abs()
                            < bands["random_common_cause_abs_delta_max"].as_f64().unwrap(),
                        "{label}: random common cause moved the ATE to {}",
                        rcc.refuted_ate
                    );
                }
            }
        }
    }
}

/// Bayesian `AverageEffect × Dag` (`bayesian.gcomp`) on explicit and accepted
/// structure under every validation level: `conformance/bayesian/shared_functional_ate`
/// (`Y = 2T + 0.5Z`) pins the posterior mean to the structural ATE; cheap/full
/// attach refuters and predictive checks, full adds prior sensitivity.
#[test]
fn average_effect_dag_bayesian_known_truth_all_structures_and_suites() {
    let expected =
        json(include_str!("../../../conformance/bayesian/shared_functional_ate/expected.json"));
    let truth = expected["true_ate"].as_f64().unwrap();
    let tolerance = expected["tolerance"].as_f64().unwrap();
    let n = 80;
    let z: Vec<f64> = (0..n).map(|i| f64::from(i) * 0.1).collect();
    let t: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let y: Vec<f64> = t.iter().zip(&z).map(|(t, z)| truth * t + 0.5 * z).collect();
    let data = TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("t", t.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let graph = dag(3, &[(0, 1), (0, 2), (1, 2)]);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
    for accepted in [false, true] {
        for suite in SUITES {
            let label = format!("accepted={accepted} suite={suite:?}");
            let result =
                staged(&data, &graph, accepted, query.clone(), bayes(400, 100.0), suite, 5);
            assert_eq!(result.logical_plan.estimator.as_deref(), Some("bayesian.gcomp"), "{label}");
            let posterior = result.posterior.as_ref().expect("Bayesian posterior");
            let mean = posterior.summaries.mean[posterior.effect_column().unwrap()];
            assert!((mean - truth).abs() < tolerance, "{label}: posterior mean {mean}");
            assert!((result.estimate.ate - mean).abs() < 1e-12, "{label}");
            assert_bayesian_validation(&result, suite, &label);
        }
    }
}

// ------------------------------------------------------------ ConditionalEffect

/// Frequentist `ConditionalEffect × Dag` on explicit and accepted structure
/// under every validation level. `conformance/context/conditional_effect`:
/// `Y = 1 + 2T + 0.5 T·W` with mean W = 2, so the modifier-averaged effect is 3.
/// Cheap runs overlap and the E-value; full refits the interaction model under
/// the effect refuters.
#[test]
fn conditional_effect_dag_frequentist_known_truth_all_structures_and_suites() {
    let expected =
        json(include_str!("../../../conformance/context/conditional_effect/expected.json"));
    let target = expected["ate_target"].as_f64().unwrap();
    let tolerance = expected["ate_tol"].as_f64().unwrap();
    let n = 200;
    let t: Vec<f64> = (0..n).map(|i| f64::from(i % 2)).collect();
    let w: Vec<f64> = (0..n).map(|i| f64::from(i % 5)).collect();
    let y: Vec<f64> = t.iter().zip(&w).map(|(t, w)| 1.0 + 2.0 * t + 0.5 * t * w).collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("w", w.as_slice()),
    ])
    .unwrap();
    let graph = dag(3, &[(0, 1), (2, 1)]);
    let query = ConditionalEffectQuery::try_new(
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_effect_modifiers([VariableId::from_raw(2)]),
    )
    .unwrap();
    for accepted in [false, true] {
        for suite in SUITES {
            let label = format!("accepted={accepted} suite={suite:?}");
            let result = staged(
                &data,
                &graph,
                accepted,
                CausalQuery::ConditionalEffect(query.clone()),
                InferenceMode::Frequentist,
                suite,
                4,
            );
            assert_eq!(
                result.logical_plan.estimator.as_deref(),
                Some("conditional.linear.adjustment"),
                "{label}"
            );
            assert!(
                (result.estimate.ate - target).abs() < tolerance,
                "{label}: {}",
                result.estimate.ate
            );
            match suite {
                RefuteSuite::None => assert!(result.refutations.is_empty(), "{label}"),
                RefuteSuite::Cheap => {
                    report(&result, "sensitivity.evalue");
                    assert!(
                        refuter_names(&result).iter().any(|n| n.starts_with("overlap")),
                        "{label}: cheap must assess overlap; got {:?}",
                        refuter_names(&result)
                    );
                }
                _ => {
                    report(&result, "sensitivity.evalue");
                    let rcc = report(&result, "random.common_cause");
                    assert!(
                        (rcc.original_ate - result.estimate.ate).abs() < 1e-9,
                        "{label}: full must refit the interaction scalar that was licensed"
                    );
                    assert!(
                        (rcc.refuted_ate - rcc.original_ate).abs() < tolerance,
                        "{label}: an irrelevant common cause moved the effect to {}",
                        rcc.refuted_ate
                    );
                    report(&result, "placebo.treatment");
                }
            }
        }
    }
}

/// `Y = 2T + 2Z ± 0.2` on a balanced `(z, t)` design.
fn mixture_data(n: usize) -> TabularData {
    let mut treatment = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    let mut confounder = Vec::with_capacity(n);
    for _ in 0..(n / 16) {
        for (z, t, count) in [(0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)] {
            for row in 0..count {
                let epsilon = if row % 2 == 0 { -0.2 } else { 0.2 };
                treatment.push(t);
                confounder.push(z);
                outcome.push(2.0 * t + 2.0 * z + epsilon);
            }
        }
    }
    TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", confounder.as_slice()),
    ])
    .unwrap()
}

/// `ConditionalEffect × Dag × graph_posterior` for both inferences under every
/// validation level. Both identified atoms recover the structural slope 2 of
/// `Y = 2T + 2Z` (the modifier `z` enters every atom's interaction model); the
/// reversed atom's mass stays unidentified at
/// `conformance/bayesian/known_truth_mixtures` `expected_unidentified_mass`
/// and is not renormalized away. Distinct adjustment certificates keep the
/// results graph-dependent even when these numerical atom effects agree.
/// Cheap/full retain each contributing atom's refutations without a scalar mix.
#[test]
fn conditional_effect_graph_posterior_known_truth_all_inferences_and_suites() {
    let pin =
        json(include_str!("../../../conformance/bayesian/known_truth_mixtures/expected.json"));
    let unidentified = pin["static_average_effect"]["expected_unidentified_mass"].as_f64().unwrap();
    let tolerance = pin["static_average_effect"]["effect_abs_tolerance"].as_f64().unwrap();
    let data = mixture_data(160);
    let query = ConditionalEffectQuery::try_new(
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_effect_modifiers([VariableId::from_raw(2)]),
    )
    .unwrap();
    for bayesian in [false, true] {
        for suite in SUITES {
            let label = format!("bayesian={bayesian} suite={suite:?}");
            let inference = if bayesian { bayes(256, 1_000.0) } else { InferenceMode::Frequentist };
            let study = Study::tabular(data.clone())
                .graph_posterior(mixture_graph_posterior())
                .query(CausalQuery::ConditionalEffect(query.clone()))
                .inference(inference)
                .refute(suite)
                .bootstrap_replicates(0)
                .build()
                .unwrap();
            let ctx = ExecutionContext::for_tests(18);
            let result = study.prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap();
            assert_eq!(result.support_status.unwrap().as_str(), "licensed", "{label}");
            assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
            assert_eq!(
                result.logical_plan.estimator.as_deref(),
                Some(if bayesian {
                    "conditional.bayesian"
                } else {
                    "conditional.linear.adjustment"
                }),
                "{label}"
            );
            assert!(result.estimate.ate.is_nan(), "{label}: scalar aggregate must be withheld");
            let structural = result.structural_response.as_ref().expect("structural atoms");
            assert!(structural.conditional_on_identified.is_none(), "{label}");
            assert!((structural.unidentified_mass - unidentified).abs() < 1e-9, "{label}");
            let values: Vec<_> = structural
                .atoms
                .iter()
                .filter_map(|atom| match atom.value {
                    Some(antecedent_core::ResponseValue::Scalar(value)) => Some(value),
                    _ => None,
                })
                .collect();
            assert_eq!(values.len(), 2, "{label}");
            for value in values {
                assert!((value - 2.0).abs() < tolerance, "{label}: atom effect {value}");
            }
            if bayesian {
                let posterior = result.posterior.as_ref().expect("Bayesian mixture posterior");
                assert!((posterior.unidentified_mass - unidentified).abs() < 1e-9, "{label}");
            } else {
                assert!(
                    result
                        .diagnostics
                        .iter()
                        .any(|d| d.message.contains(&format!("unidentified_mass={unidentified}"))),
                    "{label}: the Frequentist mixture must publish the unidentified mass"
                );
            }
            let mixed = result
                .diagnostics
                .iter()
                .any(|d| d.code.as_ref() == "refute.envelope.effect_mixture");
            if suite == RefuteSuite::None {
                assert!(result.refutations.is_empty(), "{label}");
                assert!(!mixed, "{label}");
            } else {
                assert!(!result.refutations.is_empty(), "{label}: refuters must run");
                assert!(!mixed, "{label}: graph-dependent refuters must remain per-atom");
            }
        }
    }
}

// -------------------------------------------------------------- MediationEffect

/// `conformance/estimate/staged_static_kinds`: `A = sin(.71 i)`,
/// `M = 2A + cos(1.13 i)`, `Y = 3A + 4M + .1 sin(.31 i)`.
fn mediation_scm() -> (TabularData, Dag) {
    let a: Vec<_> = (0..500).map(|i| (f64::from(i) * 0.71).sin()).collect();
    let m: Vec<_> = a.iter().enumerate().map(|(i, a)| 2.0 * a + (i as f64 * 1.13).cos()).collect();
    let y: Vec<_> = a
        .iter()
        .zip(&m)
        .enumerate()
        .map(|(i, (a, m))| 3.0 * a + 4.0 * m + 0.1 * (i as f64 * 0.31).sin())
        .collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    (data, dag(3, &[(0, 1), (0, 2), (1, 2)]))
}

/// Bayesian `MediationEffect × Dag` (`mediation.linear`) on explicit and
/// accepted structure under every validation level. The posterior mean natural
/// direct / indirect / total effects must sit on the `staged_static_kinds` truth
/// (NDE 1.8, NIE 4.8, total 6.6) and the 90% credible interval must contain it;
/// cheap/full run the mediation-native refuters (placebo mediator, random
/// common cause; full adds the data subset).
#[test]
fn mediation_bayesian_known_truth_all_structures_and_suites() {
    let pin = json(include_str!("../../../conformance/estimate/staged_static_kinds/expected.json"));
    let (data, graph) = mediation_scm();
    let control = pin["control"].as_f64().unwrap();
    let active = pin["active"].as_f64().unwrap();
    for (key, contrast) in [
        ("direct", MediationContrast::NaturalDirect),
        ("indirect", MediationContrast::NaturalIndirect),
        ("total", MediationContrast::Total),
    ] {
        let truth = pin[key].as_f64().unwrap();
        let mut query = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            Arc::from([VariableId::from_raw(1)]),
            contrast,
        );
        query.control = Intervention::set(query.treatment, Value::f64(control));
        query.active = Intervention::set(query.treatment, Value::f64(active));
        for accepted in [false, true] {
            for suite in SUITES {
                let label = format!("{key} accepted={accepted} suite={suite:?}");
                let result = staged(
                    &data,
                    &graph,
                    accepted,
                    CausalQuery::Mediation(query.clone()),
                    bayes(256, 1_000.0),
                    suite,
                    13,
                );
                assert_eq!(
                    result.logical_plan.estimator.as_deref(),
                    Some("mediation.linear"),
                    "{label}"
                );
                let posterior = result.posterior.as_ref().expect("Bayesian mediation posterior");
                assert!(
                    (result.estimate.ate - truth).abs() < 0.15,
                    "{label}: posterior mean {}",
                    result.estimate.ate
                );
                let mut draws =
                    posterior.draws.column(posterior.effect_column().unwrap()).unwrap().to_vec();
                draws.sort_by(f64::total_cmp);
                let at = |q: f64| draws[((draws.len() - 1) as f64 * q).round() as usize];
                assert!(at(0.05) <= truth && truth <= at(0.95), "{label}: 90% interval");
                match suite {
                    RefuteSuite::None => assert!(result.refutations.is_empty(), "{label}"),
                    RefuteSuite::Cheap => assert_eq!(result.refutations.len(), 2, "{label}"),
                    _ => assert_eq!(result.refutations.len(), 3, "{label}"),
                }
            }
        }
    }
}

// --------------------------------------------------- InterventionalDistribution

/// Confounded `z -> t -> y <- z` binary count table from
/// `conformance/estimate/functional_validation`: `P(y = 1 | do(t = 1)) = 0.7`.
fn distribution_table() -> TabularData {
    let mut t = Vec::new();
    let mut y = Vec::new();
    let mut z = Vec::new();
    for (zv, tv, yv, count) in [
        (0.0, 0.0, 0.0, 21),
        (0.0, 0.0, 1.0, 9),
        (0.0, 1.0, 0.0, 4),
        (0.0, 1.0, 1.0, 16),
        (1.0, 0.0, 0.0, 12),
        (1.0, 0.0, 1.0, 3),
        (1.0, 1.0, 0.0, 14),
        (1.0, 1.0, 1.0, 21),
    ] {
        t.extend(std::iter::repeat_n(tv, count));
        y.extend(std::iter::repeat_n(yv, count));
        z.extend(std::iter::repeat_n(zv, count));
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

/// Bayesian `InterventionalDistribution × Dag` (`functional.distribution`) on
/// explicit and accepted structure under every validation level. The
/// Dirichlet-posterior mean of `P(y | do(t = 1))` must sit on the
/// `functional_validation` truth 0.7, its atoms must form a coherent law, and
/// cheap/full must run the distribution subset-stability suite with the
/// fixture's report counts.
#[test]
fn distribution_bayesian_known_truth_all_structures_and_suites() {
    let pin =
        json(include_str!("../../../conformance/estimate/functional_validation/expected.json"));
    let truth = pin["distribution_mean"].as_f64().unwrap();
    let data = distribution_table();
    let graph = dag(3, &[(2, 0), (0, 1), (2, 1)]);
    let query = InterventionalDistributionQuery::new(
        VariableId::from_raw(1),
        [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
    );
    for accepted in [false, true] {
        for suite in SUITES {
            let label = format!("accepted={accepted} suite={suite:?}");
            let result = staged(
                &data,
                &graph,
                accepted,
                CausalQuery::Distribution(query.clone()),
                bayes(128, 1_000.0),
                suite,
                18,
            );
            assert_eq!(
                result.logical_plan.estimator.as_deref(),
                Some("functional.distribution"),
                "{label}"
            );
            assert!(result.posterior.is_some(), "{label}: Bayesian atoms carry draws");
            let dist = result.distribution.as_ref().unwrap();
            assert!((dist.mean - truth).abs() < 0.08, "{label}: posterior mean {}", dist.mean);
            let mass: f64 = dist.atoms.iter().map(|a| a.probability).sum();
            assert!((mass - 1.0).abs() < 1e-9, "{label}: atoms must be a law, mass={mass}");
            let expected_reports = match suite {
                RefuteSuite::None => 0,
                RefuteSuite::Cheap => pin["cheap_distribution_reports"].as_u64().unwrap(),
                _ => pin["full_distribution_reports"].as_u64().unwrap(),
            };
            assert_eq!(result.refutations.len() as u64, expected_reports, "{label}");
            assert!(
                result.refutations.iter().all(|r| r.refuter.starts_with("distribution.")),
                "{label}: {:?}",
                refuter_names(&result)
            );
        }
    }
}

// --------------------------------------------------------------- Counterfactual

/// Treatment `a` with a genuine `a × b` interaction: the true unit effect is
/// `0.8` when `b = 0` and `1.4` when `b = 1`. `z` confounds `a` and `y`.
///
/// Variables: `0 = z`, `1 = a`, `2 = b`, `3 = y`.
fn interaction_scm(binary_outcome: bool) -> (TabularData, Dag) {
    let n = 800usize;
    let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.71).sin()).collect();
    let a: Vec<f64> =
        (0..n).map(|i| f64::from(u8::from((i as f64 * 1.37).sin() + 0.8 * z[i] > 0.0))).collect();
    let b: Vec<f64> = (0..n).map(|i| f64::from(u8::from((i as f64 * 2.11).cos() > 0.0))).collect();
    let latent: Vec<f64> = (0..n)
        .map(|i| 0.8 * a[i] + 0.5 * b[i] + 0.6 * a[i] * b[i] + z[i] + 0.2 * (i as f64 * 0.29).sin())
        .collect();
    let y: Vec<f64> = if binary_outcome {
        latent.iter().map(|v| f64::from(u8::from(*v > 0.9))).collect()
    } else {
        latent
    };
    let data = TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    (data, dag(4, &[(0, 1), (0, 3), (1, 3), (2, 3)]))
}

fn counterfactual_query() -> CausalQuery {
    CausalQuery::Counterfactual(
        CounterfactualQuery::new(
            VariableId::from_raw(3),
            Arc::from([Intervention::set(VariableId::from_raw(1), Value::f64(1.0))]),
        )
        .with_control_level(0.0),
    )
}

fn homogeneity_diagnostic(result: &StudyResult) -> Option<&antecedent_core::Diagnostic> {
    result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "gcm.counterfactual.unit_effects_homogeneous")
}

fn unit_effect_spread(effects: &[f64]) -> f64 {
    effects.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - effects.iter().copied().fold(f64::INFINITY, f64::min)
}

fn diagnostic<'a>(result: &'a StudyResult, code: &str) -> Option<&'a antecedent_core::Diagnostic> {
    result.diagnostics.iter().find(|d| d.code.as_ref() == code)
}

fn field(diagnostic: &antecedent_core::Diagnostic, key: &str) -> Option<String> {
    diagnostic.fields.iter().find(|(k, _)| k.as_ref() == key).map(|(_, v)| v.to_string())
}

fn subgroup_mean(effects: &[f64], key: &[f64], want: f64) -> f64 {
    let picked: Vec<f64> =
        effects.iter().zip(key).filter(|(_, k)| **k == want).map(|(e, _)| *e).collect();
    picked.iter().sum::<f64>() / picked.len() as f64
}

/// `Counterfactual × Dag × explicit|accepted × Frequentist|Bayesian × none`:
/// the counterfactual registry offers heterogeneity-capable mechanism families
/// beside the linear one, and on this SCM — a genuine `a × b` interaction — one
/// of them wins on validation score. Abduction–action–prediction then recovers
/// the two subgroup effects (0.8 where `b = 0`, 1.4 where `b = 1`) instead of
/// one pooled slope for every unit, abduction stays exact, and nothing is
/// disclosed as homogeneous.
///
/// Before the heterogeneity families existed this cell returned the same
/// contrast for all 800 units (spread 0) and disclosed it; that is the defect
/// this pins.
#[test]
fn counterfactual_dag_interaction_outcome_recovers_subgroup_unit_effects() {
    let (data, graph) = interaction_scm(false);
    let b = data.float64_values(VariableId::from_raw(2)).unwrap();
    for accepted in [false, true] {
        for inference in [InferenceMode::Frequentist, bayes(64, 1_000.0)] {
            let bayesian = matches!(inference, InferenceMode::Bayesian(_));
            let label = format!("accepted={accepted} inference={inference:?}");
            let result = staged(
                &data,
                &graph,
                accepted,
                counterfactual_query(),
                inference,
                RefuteSuite::None,
                21,
            );
            assert_eq!(result.logical_plan.estimator.as_deref(), Some("gcm.fit"), "{label}");
            let cf = result.counterfactual.as_ref().unwrap();
            assert_eq!(cf.unit_effects.len(), 800, "{label}");

            let low = subgroup_mean(&cf.unit_effects, &b, 0.0);
            let high = subgroup_mean(&cf.unit_effects, &b, 1.0);
            assert!((low - 0.8).abs() < 0.05, "{label}: b=0 subgroup effect {low}");
            assert!((high - 1.4).abs() < 0.05, "{label}: b=1 subgroup effect {high}");
            assert!(unit_effect_spread(&cf.unit_effects) > 0.5, "{label}");

            assert!(!result.estimate.unit_effects_homogeneous, "{label}");
            assert!(homogeneity_diagnostic(&result).is_none(), "{label}");
            let mechanisms = diagnostic(&result, "gcm.counterfactual.mechanisms").unwrap();
            assert!(
                mechanisms.message.contains("LinearInteractions")
                    || mechanisms.message.contains("LinearSpline"),
                "{label}: the selection must be auditable: {}",
                mechanisms.message
            );
            let engine = diagnostic(&result, "gcm.counterfactual").unwrap();
            // `a` is categorical, so the model-wide mode is Posterior exactly as it
            // was for the linear family; the outcome's own abduction is exact
            // inversion (pinned in the model crate).
            assert!(engine.message.contains("noise_inference="), "{label}");

            if bayesian {
                let intervals = cf
                    .unit_effect_intervals
                    .as_ref()
                    .unwrap_or_else(|| panic!("{label}: Bayesian unit intervals missing"));
                assert_eq!(
                    intervals.level,
                    antecedent::result::REPORTED_SE_INTERVAL_LEVEL,
                    "{label}"
                );
                assert_eq!(intervals.method, "unit_posterior_quantile", "{label}");
                assert_eq!(intervals.lower.len(), 800, "{label}");
                for (unit, effect) in cf.unit_effects.iter().enumerate() {
                    assert!(
                        intervals.lower[unit] <= *effect && *effect <= intervals.upper[unit],
                        "{label}: unit {unit} mean {effect} outside [{}, {}]",
                        intervals.lower[unit],
                        intervals.upper[unit]
                    );
                }
                assert!(
                    diagnostic(&result, "gcm.counterfactual.uncertainty_unavailable").is_none()
                );
            } else {
                assert!(cf.unit_effect_intervals.is_none(), "{label}");
                assert!(
                    diagnostic(&result, "gcm.counterfactual.uncertainty_unavailable").is_some(),
                    "{label}: Frequentist unit effects must keep saying uncertainty is unavailable"
                );
            }
        }
    }
}

/// `y = 0.8a + z + e` has no effect modifier. The interaction families are scored
/// and lose on validation score, so the selected mechanism is the linear one and
/// every unit gets the same contrast — but that equality is a finding about the
/// data, not a property the mechanism fixed in advance, so the homogeneity
/// disclosure must not fire. What was rejected is recorded instead.
#[test]
fn counterfactual_dag_additive_outcome_reports_rejected_heterogeneity_not_homogeneity() {
    let n = 1500usize;
    let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.71).sin()).collect();
    let a: Vec<f64> =
        (0..n).map(|i| f64::from(u8::from((i as f64 * 1.37).sin() + 0.8 * z[i] > 0.0))).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 0.8 * a[i] + z[i] + 0.5 * (i as f64 * 0.29).sin() + 0.4 * (i as f64 * 2.3).cos())
        .collect();
    let data = TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("a", a.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let graph = dag(3, &[(0, 1), (0, 2), (1, 2)]);
    let query = CausalQuery::Counterfactual(
        CounterfactualQuery::new(
            VariableId::from_raw(2),
            Arc::from([Intervention::set(VariableId::from_raw(1), Value::f64(1.0))]),
        )
        .with_control_level(0.0),
    );
    let result =
        staged(&data, &graph, false, query, InferenceMode::Frequentist, RefuteSuite::None, 21);
    let cf = result.counterfactual.as_ref().unwrap();
    assert!(unit_effect_spread(&cf.unit_effects) < 1e-9, "linear family must still be selected");
    assert!(!result.estimate.unit_effects_homogeneous);
    assert!(homogeneity_diagnostic(&result).is_none());
    let rejected = diagnostic(&result, "gcm.counterfactual.heterogeneity_rejected")
        .expect("the rejected heterogeneity families must be recorded");
    assert_eq!(rejected.severity, antecedent_core::DiagnosticSeverity::Info);
    let families = field(rejected, "families").unwrap();
    assert!(families.contains("linear_interactions"), "{families}");
    assert!(rejected.message.contains("lost to linear_gaussian"), "{}", rejected.message);
}

/// A single-parent outcome cannot form any cross-parent product, so every
/// heterogeneity-capable family fails to fit and the constant contrast really is
/// fixed by the mechanism: the disclosure fires, with the structured fields the
/// consumer reads.
#[test]
fn counterfactual_dag_single_parent_outcome_discloses_structural_homogeneity() {
    let n = 800usize;
    let a: Vec<f64> = (0..n).map(|i| f64::from(u8::from((i as f64 * 1.37).sin() > 0.0))).collect();
    let y: Vec<f64> = (0..n).map(|i| 0.8 * a[i] + 0.3 * (i as f64 * 0.29).sin()).collect();
    let data = TabularData::from_f64_columns([("a", a.as_slice()), ("y", y.as_slice())]).unwrap();
    let graph = dag(2, &[(0, 1)]);
    let query = CausalQuery::Counterfactual(
        CounterfactualQuery::new(
            VariableId::from_raw(1),
            Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
        )
        .with_control_level(0.0),
    );
    for accepted in [false, true] {
        for inference in [InferenceMode::Frequentist, bayes(64, 1_000.0)] {
            let label = format!("accepted={accepted} inference={inference:?}");
            let result =
                staged(&data, &graph, accepted, query.clone(), inference, RefuteSuite::None, 21);
            let cf = result.counterfactual.as_ref().unwrap();
            assert!(unit_effect_spread(&cf.unit_effects) < 1e-9, "{label}");
            assert!(result.estimate.unit_effects_homogeneous, "{label}");
            let disclosure = homogeneity_diagnostic(&result)
                .unwrap_or_else(|| panic!("{label}: homogeneity disclosure missing"));
            assert_eq!(disclosure.severity, antecedent_core::DiagnosticSeverity::Warning);
            assert!(
                disclosure.message.contains("LinearGaussian")
                    && disclosure.message.contains("for y")
                    && disclosure.message.contains("admits no effect modification"),
                "{label}: {}",
                disclosure.message
            );
            assert_eq!(field(disclosure, "outcome").as_deref(), Some("y"), "{label}");
            assert_eq!(field(disclosure, "family").as_deref(), Some("linear_gaussian"), "{label}");
            assert!(diagnostic(&result, "gcm.counterfactual.heterogeneity_rejected").is_none());
        }
    }
}

/// `y = a·exp(z) + z + e`: the unit effect is `exp(z)`. The spline family can
/// bend the contrast with `z`, so the published unit effects must rise with `z`
/// (checked as a rank correlation) and nothing is disclosed as homogeneous.
#[test]
fn counterfactual_dag_exp_modifier_unit_effects_rise_with_the_modifier() {
    let n = 2000usize;
    let z: Vec<f64> = (0..n).map(|i| 1.2 * (i as f64 * 0.713).sin()).collect();
    let a: Vec<f64> =
        (0..n).map(|i| f64::from(u8::from((i as f64 * 1.37).sin() + 0.5 * z[i] > 0.0))).collect();
    let y: Vec<f64> =
        (0..n).map(|i| a[i] * z[i].exp() + z[i] + 0.3 * (i as f64 * 0.29).sin()).collect();
    let data = TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("a", a.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let graph = dag(3, &[(0, 1), (0, 2), (1, 2)]);
    let query = CausalQuery::Counterfactual(
        CounterfactualQuery::new(
            VariableId::from_raw(2),
            Arc::from([Intervention::set(VariableId::from_raw(1), Value::f64(1.0))]),
        )
        .with_control_level(0.0),
    );
    let result =
        staged(&data, &graph, false, query, InferenceMode::Frequentist, RefuteSuite::None, 21);
    let cf = result.counterfactual.as_ref().unwrap();
    assert!(!result.estimate.unit_effects_homogeneous);
    assert!(homogeneity_diagnostic(&result).is_none());
    let rank = |values: &[f64]| {
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&i, &j| values[i].partial_cmp(&values[j]).unwrap());
        let mut ranks = vec![0.0; values.len()];
        for (r, i) in order.into_iter().enumerate() {
            ranks[i] = r as f64;
        }
        ranks
    };
    let (rz, re) = (rank(&z), rank(&cf.unit_effects));
    let m = (n as f64 - 1.0) / 2.0;
    let cov: f64 = rz.iter().zip(&re).map(|(x, y)| (x - m) * (y - m)).sum();
    let var: f64 = rz.iter().map(|x| (x - m).powi(2)).sum();
    let spearman = cov / var;
    assert!(spearman > 0.95, "unit effects must rise with z; spearman={spearman}");
    // And the level is right where the data are dense: exp(0) = 1 at z ≈ 0.
    let near_zero: Vec<f64> =
        cf.unit_effects.iter().zip(&z).filter(|(_, zi)| zi.abs() < 0.1).map(|(e, _)| *e).collect();
    let centre = near_zero.iter().sum::<f64>() / near_zero.len() as f64;
    assert!((centre - 1.0).abs() < 0.1, "unit effect near z=0 is {centre}");
}

/// Per-unit support: treated units whose covariate `x` exceeds every control
/// unit's `x` are predicted into the control arm at a cell the fit never saw.
/// The pooled treatment range is fully observed (`extrapolative=false`), so only
/// the per-unit flag can say so — and it must flag exactly those units.
#[test]
fn counterfactual_dag_flags_units_outside_the_opposite_arms_covariate_support() {
    let n = 1200usize;
    let a: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
    // Controls see x in [0, 1); treated units reach up to 1.5.
    let x: Vec<f64> = (0..n)
        .map(|i| {
            let u = ((i as f64 * 0.618_033_988_7) % 1.0).abs();
            if a[i] > 0.5 { 1.5 * u } else { u }
        })
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 0.5 * a[i] + x[i] + 0.7 * a[i] * x[i] + 0.2 * (i as f64 * 0.29).sin())
        .collect();
    let data = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("a", a.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let graph = dag(3, &[(0, 2), (1, 2)]);
    let query = CausalQuery::Counterfactual(
        CounterfactualQuery::new(
            VariableId::from_raw(2),
            Arc::from([Intervention::set(VariableId::from_raw(1), Value::f64(1.0))]),
        )
        .with_control_level(0.0),
    );
    let result =
        staged(&data, &graph, false, query, InferenceMode::Frequentist, RefuteSuite::None, 21);
    let cf = result.counterfactual.as_ref().unwrap();
    let flags = cf.unit_extrapolative.as_ref().expect("per-unit support flags");
    let arm_range = |treated: bool| {
        x.iter()
            .zip(&a)
            .filter(|(_, ai)| (**ai > 0.5) == treated)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (xi, _)| {
                (lo.min(*xi), hi.max(*xi))
            })
    };
    let (control, treated) = (arm_range(false), arm_range(true));
    let mut expected_cells = 0usize;
    let mut above_control = 0usize;
    for unit in 0..n {
        let (lo, hi) = if a[unit] > 0.5 { control } else { treated };
        let outside_cell = x[unit] < lo || x[unit] > hi;
        above_control += usize::from(a[unit] > 0.5 && x[unit] > control.1);
        if outside_cell {
            expected_cells += 1;
            assert!(flags[unit], "unit {unit} (a={}, x={}) must be flagged", a[unit], x[unit]);
        }
    }
    assert!(above_control > 100, "the fixture must put real treated mass above control support");
    // Nothing else is flagged for the covariate cell.
    let cell_flags = flags.iter().filter(|f| **f).count();
    assert!(cell_flags >= expected_cells);
    let support = diagnostic(&result, "gcm.counterfactual.support").unwrap();
    assert_eq!(field(support, "extrapolative").as_deref(), Some("false"));
    assert_eq!(field(support, "parent_cell_extrapolative"), Some(expected_cells.to_string()));
    assert_eq!(support.severity, antecedent_core::DiagnosticSeverity::Warning);
    assert!(support.message.contains("covariates=[x]"), "{}", support.message);
}

/// A parent-conditional categorical lever whose parents sit on very different
/// scales (a raw year column beside a 0–1 axis). The Bayesian refit used to run
/// the multinomial IRLS on the raw columns under Dirichlet row weights and stop
/// short of tolerance, refusing the whole analysis; standardizing parents
/// inside the fit makes the counterfactual identical under an affine rescaling
/// of those parents, on either inference mode.
#[test]
fn counterfactual_dag_categorical_lever_is_invariant_to_parent_rescaling() {
    let n = 3000usize;
    let frac = |i: usize, k: f64| ((i as f64 * k) % 1.0).abs();
    let year: Vec<f64> =
        (0..n).map(|i| 2015.0 + (frac(i, 0.618_033_988_7) * 11.0).floor()).collect();
    let axis: Vec<f64> = (0..n).map(|i| frac(i, 0.414_213_562_4)).collect();
    let lever: Vec<f64> = (0..n)
        .map(|i| {
            let eta = 0.4 * (year[i] - 2020.0) / 3.16 + axis[i] - 0.5;
            f64::from(u8::from(frac(i, 0.732_050_807_6) < 1.0 / (1.0 + (-eta).exp())))
        })
        .collect();
    let y: Vec<f64> =
        (0..n).map(|i| 0.06 * lever[i] + 0.3 * axis[i] + 0.5 * (i as f64 * 0.29).sin()).collect();
    let standardize = |col: &[f64]| {
        let mean = col.iter().sum::<f64>() / n as f64;
        let sd = (col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        col.iter().map(|v| (v - mean) / sd).collect::<Vec<f64>>()
    };
    let (year_std, axis_std) = (standardize(&year), standardize(&axis));
    let graph = dag(4, &[(0, 2), (1, 2), (2, 3), (1, 3)]);
    let query = CausalQuery::Counterfactual(
        CounterfactualQuery::new(
            VariableId::from_raw(3),
            Arc::from([Intervention::set(VariableId::from_raw(2), Value::f64(1.0))]),
        )
        .with_control_level(0.0),
    );
    for inference in [InferenceMode::Frequentist, bayes(32, 1_000.0)] {
        let label = format!("{inference:?}");
        let run = |year_col: &[f64], axis_col: &[f64]| {
            let data = TabularData::from_f64_columns([
                ("year", year_col),
                ("axis", axis_col),
                ("lever", lever.as_slice()),
                ("y", y.as_slice()),
            ])
            .unwrap();
            staged(&data, &graph, false, query.clone(), inference.clone(), RefuteSuite::None, 7)
        };
        let raw = run(&year, &axis);
        let std = run(&year_std, &axis_std);
        let (raw_cf, std_cf) = (raw.counterfactual.unwrap(), std.counterfactual.unwrap());
        assert!((raw_cf.mean_ite - std_cf.mean_ite).abs() < 1e-8, "{label}");
        for (unit, (r, s)) in raw_cf.unit_effects.iter().zip(std_cf.unit_effects.iter()).enumerate()
        {
            assert!((r - s).abs() < 1e-8, "{label}: unit {unit}: raw {r} vs standardized {s}");
        }
    }
}

/// The same SCM with a binary outcome selects a parent-conditional discrete
/// mechanism, which *can* modify the effect. Nothing is disclosed and the unit
/// effects really do differ, so the disclosure is not vacuous.
#[test]
fn counterfactual_dag_discrete_outcome_makes_no_homogeneity_claim() {
    let (data, graph) = interaction_scm(true);
    let result = staged(
        &data,
        &graph,
        false,
        counterfactual_query(),
        InferenceMode::Frequentist,
        RefuteSuite::None,
        21,
    );
    let cf = result.counterfactual.as_ref().unwrap();
    assert!(
        unit_effect_spread(&cf.unit_effects) > 1e-9,
        "a discrete outcome must not produce one constant contrast"
    );
    assert!(!result.estimate.unit_effects_homogeneous);
    assert!(homogeneity_diagnostic(&result).is_none());
}

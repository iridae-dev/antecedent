//! Builder independent execution evidence for the certified trial-to-target transport routes.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::RetainedOperation;
use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, EstimatorId, IdentifierId, InferenceMode,
    RefuteSuite, Study, TransportTrialSpec,
};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    TransportQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, Dag, DenseNodeId};
use antecedent_io::consume_analysis_result;

const NAMES: [&str; 5] = ["a", "y", "trial", "s", "e"];

fn table(y: &[f64]) -> TabularData {
    TabularData::from_f64_columns([
        (NAMES[0], &[1.0, 0.0, 0.0, 0.0][..]),
        (NAMES[1], y),
        (NAMES[2], &[1.0, 1.0, 0.0, 0.0][..]),
        (NAMES[3], &[0.5, 0.5, 0.5, 0.5][..]),
        (NAMES[4], &[0.5, 0.5, 0.5, 0.5][..]),
    ])
    .unwrap()
}

fn widened() -> TabularData {
    TabularData::from_f64_columns([
        (NAMES[0], &[1.0, 0.0, 0.0, 0.0][..]),
        (NAMES[1], &[3.0, 1.0, 0.0, 0.0][..]),
        (NAMES[2], &[1.0, 1.0, 0.0, 0.0][..]),
        (NAMES[3], &[0.5, 0.5, 0.5, 0.5][..]),
        (NAMES[4], &[0.5, 0.5, 0.5, 0.5][..]),
        ("extra", &[0.0, 0.0, 0.0, 0.0][..]),
    ])
    .unwrap()
}

fn query() -> TransportQuery {
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    });
    TransportQuery::new(response, "trial", "target", [VariableId::from_raw(0)])
}

fn build(data: &TabularData, bayesian: bool) -> Study {
    let mut admg = Admg::with_variables(5);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let study = Study::tabular(data.clone())
        .graph(admg)
        .query(CausalQuery::Transport(query()))
        .selection_targets(Arc::from([]))
        .transport_trial(TransportTrialSpec {
            trial: VariableId::from_raw(2),
            selection_probability: VariableId::from_raw(3),
            treatment_probability: VariableId::from_raw(4),
        })
        .refute(RefuteSuite::None);
    let study = if bayesian {
        study.inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(2_000)))
    } else {
        study
    };
    study.build().unwrap()
}

#[test]
fn transport_trial_executes_from_retained_sid_proof_after_builder_drop() {
    run_on_large_stack(transport_trial_executes_from_retained_sid_proof_after_builder_drop_body);
}

fn transport_trial_executes_from_retained_sid_proof_after_builder_drop_body() {
    let ipw_pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/staged_transport/expected.json"
    ))
    .unwrap();
    let bayesian_pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/bayesian_trial_transport/expected.json"
    ))
    .unwrap();
    let ipw = ipw_pin["ipw"].as_f64().unwrap();
    let ipw_tolerance = ipw_pin["tolerance"].as_f64().unwrap();
    let data = table(&[3.0, 1.0, 0.0, 0.0]);
    // Shifting every outcome leaves the IPW contrast of the two trial rows unchanged.
    let shifted = table(&[3.5, 1.5, 0.5, 0.5]);

    for bayesian in [false, true] {
        let ctx = ExecutionContext::for_tests(41);
        let builder = build(&data, bayesian);
        let one_shot = builder.run(&ctx).unwrap();
        let mut prepared = builder.prepare(&ctx).unwrap();
        drop(builder);

        let expected_estimator = if bayesian {
            EstimatorId::TransportTrialBayesianBootstrap
        } else {
            EstimatorId::TransportTrialIpw
        };
        let RetainedOperation::TransportTrial(plan) = prepared.retained_operation() else {
            panic!("retained transport plan")
        };
        assert_eq!(plan.query, query());
        assert_eq!(plan.identifier, IdentifierId::TransportSid);
        assert_eq!(plan.estimator, expected_estimator);
        assert_eq!(plan.posterior_draws, bayesian.then_some(2_000));
        assert!(!plan.rule.is_empty());
        assert_eq!(prepared.plan().record.plan_id, plan.plan_id);

        let result = prepared.estimate(&data, &ctx).unwrap();
        assert!(result.diagnostics.iter().any(|item| item.code.as_ref() == "exec.identify.cached"));
        assert_eq!(result.estimand.method.as_ref(), "transport.sid.direct");
        assert_eq!(result.logical_plan.estimator.as_deref(), Some(expected_estimator.as_str()));
        let transported = result.transport().expect("transport estimate");
        if bayesian {
            let posterior = result.posterior.as_ref().expect("transport posterior");
            assert_eq!(posterior.draws.n_draws, 2_000);
            assert!(
                (posterior.summaries.mean[0] - bayesian_pin["expected_mean"].as_f64().unwrap())
                    .abs()
                    < bayesian_pin["mean_tolerance"].as_f64().unwrap()
            );
            assert!(
                (posterior.summaries.sd[0].powi(2)
                    - bayesian_pin["expected_variance"].as_f64().unwrap())
                .abs()
                    < bayesian_pin["variance_tolerance"].as_f64().unwrap()
            );
            assert!(
                (result.estimate().as_effect().unwrap().ate - posterior.summaries.mean[0]).abs()
                    < 1e-12
            );
            // The one-shot facade executes the same retained row-law bootstrap and seed.
            assert_eq!(
                one_shot.posterior.as_ref().unwrap().draws.values.as_ref(),
                posterior.draws.values.as_ref()
            );
        } else {
            assert!((transported.ipw - ipw).abs() <= ipw_tolerance);
            assert!((result.estimate().as_effect().unwrap().ate - ipw).abs() <= ipw_tolerance);
        }
        assert!(
            (one_shot.estimate().as_effect().unwrap().ate
                - result.estimate().as_effect().unwrap().ate)
                .abs()
                < 1e-12
        );

        let bytes =
            prepared.encode_contracted_result(&result, "checked-transport-trial", &ctx).unwrap();
        let consumed = consume_analysis_result(&bytes).unwrap();
        assert!(
            consumed.acceptance.unresolved.iter().any(|dependency| {
                dependency.as_ref() == "dependencies.checked_transport_trial_operation"
            }),
            "{:?}",
            consumed.acceptance.unresolved
        );
        assert!(!consumed.acceptance.accepts_as_verified_program());

        let refreshed = rebind_estimate(&mut prepared, shifted.clone(), &ctx).unwrap();
        assert!(
            refreshed.diagnostics.iter().any(|item| item.code.as_ref() == "exec.identify.cached")
        );
        assert_eq!(refreshed.estimand.method.as_ref(), "transport.sid.direct");
        if !bayesian {
            assert!((refreshed.transport().unwrap().ipw - ipw).abs() <= ipw_tolerance);
        }
        assert!(matches!(prepared.retained_operation(), RetainedOperation::TransportTrial(..)));
        assert!(rebind_estimate(&mut prepared, widened(), &ctx).is_err());
    }
}

/// Debug builds of the shared prepared dispatcher carry frames close to the
/// default test-thread stack, so the evidence body runs on its own thread.
fn run_on_large_stack(run: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("checked-route-evidence".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(run)
        .expect("spawn evidence thread")
        .join()
        .expect("evidence body panicked");
}

fn trial_spec() -> TransportTrialSpec {
    TransportTrialSpec {
        trial: VariableId::from_raw(2),
        selection_probability: VariableId::from_raw(3),
        treatment_probability: VariableId::from_raw(4),
    }
}

fn dag_study(
    data: &TabularData,
    accepted: bool,
    bayesian: bool,
    suite: RefuteSuite,
) -> antecedent::Study {
    let mut dag = Dag::with_variables(5);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let builder = Study::tabular(data.clone());
    let builder =
        if accepted { builder.graph(AcceptedGraph::from(dag)) } else { builder.graph(dag) };
    builder
        .query(CausalQuery::Transport(query()))
        .selection_targets(Arc::from([]))
        .transport_trial(trial_spec())
        .inference(if bayesian {
            InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(2_000))
        } else {
            InferenceMode::Frequentist
        })
        .refute(suite)
        .build()
        .unwrap()
}

/// A DAG is the selection diagram with no latent confounding, so the transported contrast on
/// it is the one the same diagram written as an ADMG gives, and the hand-computed value of the
/// two-row design: `(+2 * 3 - 2 * 1) / 2 = 2`.
#[test]
fn a_dag_is_the_confounder_free_selection_diagram_of_a_transport_query() {
    run_on_large_stack(|| {
        let ipw_pin: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/estimate/staged_transport/expected.json"
        ))
        .unwrap();
        let ipw = ipw_pin["ipw"].as_f64().unwrap();
        let data = table(&[3.0, 1.0, 0.0, 0.0]);
        let ctx = ExecutionContext::for_tests(43);
        for accepted in [false, true] {
            for bayesian in [false, true] {
                let built = dag_study(&data, accepted, bayesian, RefuteSuite::None);
                // No evidence row names the cell: it executes and stays unlicensed.
                assert_eq!(built.support_status(), Some(CellStatus::Refused));
                let result = built.run(&ctx).unwrap();
                assert_eq!(result.support_status(), Some(CellStatus::Refused));
                let reference = build(&data, bayesian).run(&ctx).unwrap();
                let (got, want) = (
                    result.estimate().as_effect().unwrap().ate,
                    reference.estimate().as_effect().unwrap().ate,
                );
                if bayesian {
                    // The same seeded Dirichlet draws on the same rows.
                    assert!((got - want).abs() < 1e-12, "{got} vs {want}");
                } else {
                    assert!((got - ipw).abs() < 1e-12, "{got}");
                    assert!((result.transport().unwrap().ipw - ipw).abs() < 1e-12);
                }
                assert_eq!(result.estimand.method.as_ref(), "transport.sid.direct");
                // The prepared handle freezes the proof and retains the checked operation.
                let prepared = built.prepare(&ctx).unwrap();
                drop(built);
                let RetainedOperation::TransportTrial(plan) = prepared.retained_operation() else {
                    panic!("retained transport plan")
                };
                assert_eq!(plan.identifier, IdentifierId::TransportSid);
                let again = prepared.estimate(&data, &ctx).unwrap();
                assert!(
                    again
                        .diagnostics
                        .iter()
                        .any(|item| item.code.as_ref() == "exec.identify.cached")
                );
                assert!((again.estimate().as_effect().unwrap().ate - got).abs() < 1e-12);
            }
        }
        // A CPDAG or PAG stands for several selection diagrams and still refuses.
        for class in ["Cpdag", "Pag"] {
            let status = antecedent::support::classify(
                antecedent::SupportCell::from_labels(
                    "TransportQuery",
                    class,
                    "explicit",
                    "Frequentist",
                    "none",
                )
                .unwrap(),
            );
            assert!(matches!(status, CellStatus::Refused), "{class}: {status:?}");
        }
    });
}

/// A deterministic trial of `rows` units: a fair trial coin, a fair arm coin, `y = 2 a + noise`.
/// `selection` and `propensity` are the probabilities the caller supplies.
fn coin_trial(rows: usize, selection: f64, propensity: f64) -> (TabularData, Vec<f64>) {
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut uniform = move || {
        state =
            state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    };
    let (mut a, mut y, mut trial) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..rows {
        let member = uniform() < 0.5;
        let treated = uniform() < 0.5;
        a.push(f64::from(u8::from(treated)));
        y.push(2.0 * f64::from(u8::from(treated)) + 0.3 * (uniform() - 0.5));
        trial.push(f64::from(u8::from(member)));
    }
    let s = vec![selection; rows];
    let e = vec![propensity; rows];
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("y", y.as_slice()),
        ("trial", trial.as_slice()),
        ("s", s.as_slice()),
        ("e", e.as_slice()),
    ])
    .unwrap();
    (data, [a, y, trial].concat())
}

/// The trial-to-target contrast counted by hand: each trial row contributes
/// `+-(1 - s) / (s * arm) * y`, and the sum is divided by the number of target rows.
fn counted_ipw(columns: &[f64], rows: usize, selection: f64, propensity: f64) -> f64 {
    let (a, y, trial) = (&columns[..rows], &columns[rows..2 * rows], &columns[2 * rows..]);
    let targets = trial.iter().filter(|&&member| member == 0.0).count() as f64;
    let total: f64 = (0..rows)
        .filter(|&row| trial[row] == 1.0)
        .map(|row| {
            let (sign, arm) =
                if a[row] == 1.0 { (1.0, propensity) } else { (-1.0, 1.0 - propensity) };
            sign * (1.0 - selection) / (selection * arm) * y[row]
        })
        .sum();
    total / targets
}

#[test]
fn cheap_and_full_validation_test_the_supplied_probabilities_against_the_rows() {
    run_on_large_stack(|| {
        let rows = 400;
        let (data, columns) = coin_trial(rows, 0.5, 0.5);
        let counted = counted_ipw(&columns, rows, 0.5, 0.5);
        assert!((counted - 2.0).abs() < 0.5, "{counted}");
        let trial_rows = columns[2 * rows..].iter().filter(|&&member| member == 1.0).count();
        let ctx = ExecutionContext::for_tests(47);
        for (graph_is_dag, bayesian) in [(true, false), (true, true), (false, false), (false, true)]
        {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let built = if graph_is_dag {
                    dag_study(&data, false, bayesian, suite)
                } else {
                    let mut admg = Admg::with_variables(5);
                    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
                        .unwrap();
                    Study::tabular(data.clone())
                        .graph(admg)
                        .query(CausalQuery::Transport(query()))
                        .selection_targets(Arc::from([]))
                        .transport_trial(trial_spec())
                        .inference(if bayesian {
                            InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(2_000))
                        } else {
                            InferenceMode::Frequentist
                        })
                        .refute(suite)
                        .build()
                        .unwrap()
                };
                let result = built.run(&ctx).unwrap_or_else(|e| panic!("{suite:?}: {e}"));
                let names: Vec<&str> =
                    result.refutations.iter().map(|r| r.refuter.as_ref()).collect();
                let expected: Vec<&str> = match suite {
                    RefuteSuite::None => Vec::new(),
                    RefuteSuite::Cheap => vec![
                        "transport.selection_calibration",
                        "transport.arm_calibration",
                        "transport.selection_overlap",
                        "transport.treatment_overlap",
                    ],
                    _ => vec![
                        "transport.selection_calibration",
                        "transport.arm_calibration",
                        "transport.selection_overlap",
                        "transport.treatment_overlap",
                        "transport.bootstrap_range",
                    ],
                };
                assert_eq!(names, expected, "dag={graph_is_dag} bayesian={bayesian} {suite:?}");
                let effect = result.estimate().as_effect().unwrap().ate;
                if bayesian {
                    assert!((effect - counted).abs() < 0.15, "{effect} vs {counted}");
                } else {
                    assert!((effect - counted).abs() < 1e-9, "{effect} vs {counted}");
                }
                if suite == RefuteSuite::None {
                    continue;
                }
                // The selection check compares the trial count with the sum of the supplied
                // probabilities; the decision is the two-sided 5% normal test.
                let selection = &result.refutations[0];
                assert!((selection.original_ate - trial_rows as f64).abs() < 1e-12);
                assert!((selection.refuted_ate - rows as f64 * 0.5).abs() < 1e-9);
                let z = (trial_rows as f64 - rows as f64 * 0.5) / (rows as f64 * 0.25).sqrt();
                assert_eq!(selection.passed, z.abs() < 1.959_964, "z = {z}");
                assert!(selection.informative);
                // Among the trial rows the arm count is tested against the supplied propensity.
                let treated = columns[..rows]
                    .iter()
                    .zip(&columns[2 * rows..])
                    .filter(|(arm, member)| **arm == 1.0 && **member == 1.0)
                    .count();
                let arm = &result.refutations[1];
                assert!((arm.original_ate - treated as f64).abs() < 1e-12);
                assert!((arm.refuted_ate - trial_rows as f64 * 0.5).abs() < 1e-9);
                let z =
                    (treated as f64 - trial_rows as f64 * 0.5) / (trial_rows as f64 * 0.25).sqrt();
                assert_eq!(arm.passed, z.abs() < 1.959_964, "z = {z}");
                // Fair coins give weights of equal size: the effective share is one.
                for overlap in &result.refutations[2..4] {
                    assert!(overlap.passed && overlap.comparison > 0.9, "{overlap:?}");
                }
                if suite == RefuteSuite::Full {
                    let range = &result.refutations[4];
                    assert!(range.replicates >= 40, "{}", range.replicates);
                    assert!(range.passed, "{range:?}");
                    assert!(!range.informative);
                }
            }
        }
    });
}

#[test]
fn probabilities_that_do_not_describe_the_rows_fail_their_calibration() {
    run_on_large_stack(|| {
        let rows = 400;
        let ctx = ExecutionContext::for_tests(53);
        // The trial coin and the arm coin are fair; the caller claims 0.9 for each.
        let (data, _) = coin_trial(rows, 0.9, 0.9);
        let result = dag_study(&data, false, false, RefuteSuite::Cheap).run(&ctx).unwrap();
        let by_name = |name: &str| {
            result.refutations.iter().find(|r| r.refuter.as_ref() == name).unwrap().clone()
        };
        let selection = by_name("transport.selection_calibration");
        assert!(!selection.passed, "{selection:?}");
        assert!(selection.comparison < 1e-6, "{}", selection.comparison);
        assert!(selection.refuted_ate > selection.original_ate + 100.0);
        let arm = by_name("transport.arm_calibration");
        assert!(!arm.passed, "{arm:?}");
        assert!(arm.comparison < 1e-6, "{}", arm.comparison);
    });
}

/// Bind `data` on a new branch, materialize and estimate it, then adopt the branch as the handle.
/// A failure leaves the handle unchanged, as the immutable lifecycle does.
fn rebind_estimate(
    prepared: &mut antecedent::PreparedStudy,
    data: antecedent_data::TabularData,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    let branch = prepared.rebind_data(data, ctx)?.materialize(ctx)?;
    let result = branch.estimate_retained(ctx)?;
    *prepared = branch;
    Ok(result)
}

/// Declared observation weights, clusters and series on the trial-to-target contrast, against
/// the exact rational references frozen in the estimator crate's oracle data
/// (`transport_design_reference.json`, formed by `transport_design_reference.py`).
mod declared_design {
    use std::sync::Arc;

    use antecedent::{
        BayesianConfig, CausalError, InferenceMode, RefuteSuite, ScoreDependence, Study,
        StudyBuilder, StudyResult, TransportTrialSpec,
    };
    use antecedent_core::{
        CausalQuery, DeclaredSamplingDesign, ExecutionContext, VariableId,
    };
    use antecedent_data::TabularData;
    use antecedent_graph::{Admg, DenseNodeId};
    use serde_json::Value;

    fn reference() -> Value {
        serde_json::from_str(include_str!(
            "../../antecedent-estimate/tests/oracle_data/transport_design_reference.json"
        ))
        .unwrap()
    }

    fn floats(v: &Value) -> Vec<f64> {
        v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
    }

    fn text(v: &Value) -> f64 {
        v.as_str().unwrap().parse().unwrap()
    }

    fn labels(v: &Value) -> Vec<u32> {
        v.as_array().unwrap().iter().map(|x| u32::try_from(x.as_u64().unwrap()).unwrap()).collect()
    }

    /// The case's columns `a, y, trial, s, e` as a table, with its weights when it has them.
    fn table(rows: &Value) -> TabularData {
        let flag = |v: &Value| -> Vec<f64> {
            v.as_array().unwrap().iter().map(|x| f64::from(u8::from(x.as_bool().unwrap()))).collect()
        };
        let (a, trial) = (flag(&rows["treatment"]), flag(&rows["trial"]));
        let (y, s, e) =
            (floats(&rows["outcome"]), floats(&rows["selection"]), floats(&rows["propensity"]));
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("y", y.as_slice()),
            ("trial", trial.as_slice()),
            ("s", s.as_slice()),
            ("e", e.as_slice()),
        ])
        .unwrap();
        match rows["weights"].is_null() {
            true => data,
            false => data.with_weights(Arc::from(floats(&rows["weights"]))).unwrap(),
        }
    }

    fn builder(data: TabularData) -> StudyBuilder {
        let mut admg = Admg::with_variables(5);
        admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        Study::tabular(data)
            .graph(admg)
            .query(CausalQuery::Transport(super::query()))
            .selection_targets(Arc::from([]))
            .transport_trial(TransportTrialSpec {
                trial: VariableId::from_raw(2),
                selection_probability: VariableId::from_raw(3),
                treatment_probability: VariableId::from_raw(4),
            })
            .refute(RefuteSuite::None)
    }

    fn run(builder: StudyBuilder) -> Result<StudyResult, CausalError> {
        builder.build()?.run(&ExecutionContext::for_tests(61))
    }

    fn declare(builder: StudyBuilder, case: &Value) -> StudyBuilder {
        match case["dependence"]["kind"].as_str().unwrap() {
            "clusters" => builder
                .sampling_design(DeclaredSamplingDesign::ClusteredPanel)
                .score_dependence(ScoreDependence::clusters(labels(&case["dependence"]["labels"]))),
            "series" => builder
                .sampling_design(DeclaredSamplingDesign::DependentTimeSeries)
                .score_dependence(ScoreDependence::newey_west(
                    usize::try_from(case["dependence"]["lag"].as_u64().unwrap()).unwrap(),
                )),
            _ => builder,
        }
    }

    fn close(got: f64, want: f64, what: &str) {
        assert!((got - want).abs() <= 1e-10 * want.abs().max(1.0), "{what}: {got} vs {want}");
    }

    fn on_large_stack(run: impl FnOnce() + Send + 'static) {
        super::run_on_large_stack(run);
    }

    #[test]
    fn transport_weights_clusters_and_series_publish_the_exact_reference_point_and_error() {
        on_large_stack(|| {
            for case in reference()["estimate"].as_array().unwrap() {
                let name = case["name"].as_str().unwrap();
                let result = run(declare(builder(table(&case["rows"])), case)).unwrap();
                let effect = result.estimate().as_effect().unwrap();
                close(effect.ate, text(&case["ipw"]), &format!("{name} point"));
                close(result.transport().unwrap().ipw, text(&case["ipw"]), name);
                close(effect.uncertainty.analytic.se, text(&case["se"]), &format!("{name} se"));
                // The error kind names what was read.
                let kind = effect.uncertainty.analytic.kind;
                match case["dependence"]["kind"].as_str().unwrap() {
                    "clusters" => {
                        assert_eq!(kind, Some(antecedent_estimate::AnalyticSeKind::Cluster), "{name}");
                        assert!(effect.uncertainty.analytic.reference_df.is_some(), "{name}");
                    }
                    "series" => assert!(
                        matches!(kind, Some(antecedent_estimate::AnalyticSeKind::NeweyWest { .. })),
                        "{name}: {kind:?}"
                    ),
                    _ => {}
                }
            }
        });
    }

    #[test]
    fn transport_bayesian_bootstrap_follows_the_exact_law_over_weighted_rows_clusters_and_blocks() {
        on_large_stack(|| {
            let draws = 60_000;
            for case in reference()["bayes"].as_array().unwrap() {
                let name = case["name"].as_str().unwrap();
                let mut study = builder(table(&case["rows"]))
                    .inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(draws)));
                study = match case["grouping"].as_str().unwrap() {
                    "clusters" => study
                        .sampling_design(DeclaredSamplingDesign::ClusteredPanel)
                        .score_dependence(ScoreDependence::clusters(labels(&case["labels"]))),
                    "blocks" => {
                        let block = usize::try_from(case["block"].as_u64().unwrap()).unwrap();
                        study
                            .sampling_design(DeclaredSamplingDesign::DependentTimeSeries)
                            .score_dependence(ScoreDependence::newey_west(block))
                    }
                    _ => study,
                };
                let result = run(study).unwrap();
                let posterior = result.posterior.as_ref().expect("transport posterior");
                let values = posterior.draws.values.as_ref();
                let n = values.len() as f64;
                assert_eq!(values.len(), draws, "{name}");
                for point in case["cdf"].as_array().unwrap() {
                    let (x, p) = (text(&point["x"]), text(&point["p"]));
                    let empirical = values.iter().filter(|v| **v <= x).count() as f64 / n;
                    let tol = 5.0 * (p * (1.0 - p) / n).sqrt();
                    assert!((empirical - p).abs() < tol, "{name} F({x}) = {empirical} vs {p}");
                }
            }
        });
    }

    #[test]
    fn transport_series_without_a_lag_takes_the_default_rule_on_any_graph() {
        on_large_stack(|| {
            let all = reference();
            let case = &all["estimate"].as_array().unwrap()[1];
            let n = case["rows"]["outcome"].as_array().unwrap().len();
            let lag = antecedent_estimate::RowResampling::SeriesBlocks.block_length(n).unwrap();
            let by_default = run(builder(table(&case["rows"]))
                .sampling_design(DeclaredSamplingDesign::DependentTimeSeries))
            .unwrap();
            let explicit = run(builder(table(&case["rows"]))
                .sampling_design(DeclaredSamplingDesign::DependentTimeSeries)
                .score_dependence(ScoreDependence::newey_west(lag)))
            .unwrap();
            let se = |r: &StudyResult| r.estimate().as_effect().unwrap().uncertainty.analytic.se;
            assert!((se(&by_default) - se(&explicit)).abs() < 1e-12);
            let independent = run(builder(table(&case["rows"]))).unwrap();
            assert!((se(&by_default) - se(&independent)).abs() > 1e-6);
        });
    }

    #[test]
    fn transport_validation_runs_on_clusters_and_is_refused_beside_weights() {
        on_large_stack(|| {
            let estimate = reference()["estimate"].as_array().unwrap().clone();
            let find = |name: &str| estimate.iter().find(|c| c["name"] == name).unwrap().clone();
            let clustered = find("clusters_unbalanced_unweighted");
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let result =
                    run(declare(builder(table(&clustered["rows"])), &clustered).refute(suite))
                        .unwrap();
                close(
                    result.estimate().as_effect().unwrap().uncertainty.analytic.se,
                    text(&clustered["se"]),
                    "cluster se under validation",
                );
                assert!(
                    result
                        .refutations
                        .iter()
                        .any(|r| r.refuter.as_ref() == "transport.selection_calibration"),
                    "{suite:?}"
                );
                if suite == RefuteSuite::Full {
                    let range = result
                        .refutations
                        .iter()
                        .find(|r| r.refuter.as_ref() == "transport.bootstrap_range")
                        .expect("a whole-cluster bootstrap range");
                    assert!(range.replicates >= 2, "{range:?}");
                }
            }
            // The unweighted suite would describe a different population than the weighted
            // number: refused, not run.
            let weighted = find("independent_weighted");
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let error =
                    run(builder(table(&weighted["rows"])).refute(suite)).err().expect("refused");
                assert_eq!(error.reason_code(), Some("option_not_applicable"), "{error}");
                assert!(error.to_string().contains("observation weights"), "{error}");
            }
        });
    }

    #[test]
    fn transport_declarations_the_route_would_ignore_are_refused_not_dropped() {
        on_large_stack(|| {
            let all = reference();
            let base = &all["estimate"].as_array().unwrap()[0];
            let plain = run(builder(table(&base["rows"]))).unwrap();
            let refused = |result: Result<StudyResult, CausalError>, needle: &str| {
                let error = result.err().expect("a declaration was ignored");
                assert_eq!(error.reason_code(), Some("option_not_applicable"), "{error}");
                assert!(error.to_string().contains(needle), "{error}");
                assert!(error.to_string().contains("Next attempt"), "{error}");
            };
            // An estimator the route does not run: it would be answered by weighting.
            refused(
                run(builder(table(&base["rows"])).estimator(antecedent::EstimatorId::Aipw)),
                "inverse-probability weighting",
            );
            // The route's own estimator, named, is the plain run.
            let named = run(
                builder(table(&base["rows"])).estimator(antecedent::EstimatorId::TransportTrialIpw),
            )
            .unwrap();
            assert_eq!(
                named.estimate().as_effect().unwrap().ate,
                plain.estimate().as_effect().unwrap().ate
            );
            // Clip or trim of probabilities that are the design, and a population registry the
            // target rows never consult.
            refused(
                run(builder(table(&base["rows"])).overlap_policy(
                    antecedent_estimate::OverlapPolicy::RequireDiagnostics {
                        clip: Some(0.2),
                        trim: Some(0.2),
                    },
                )),
                "overlap policy",
            );
            refused(
                run(builder(table(&base["rows"]))
                    .population_registry(antecedent_core::PopulationRegistry::default())),
                "population registry",
            );
            // The replicate count has a default, so it cannot be told from a declaration; the
            // full suite is the one that reads it, in its bootstrap range.
            let full = run(builder(table(&base["rows"]))
                .bootstrap_replicates(50)
                .refute(RefuteSuite::Full))
            .unwrap();
            let range = full
                .refutations
                .iter()
                .find(|r| r.refuter.as_ref() == "transport.bootstrap_range")
                .expect("the bootstrap range");
            assert!(range.replicates + 20 >= 50, "{range:?}");
        });
    }

    #[test]
    fn transport_declarations_the_rows_cannot_honour_are_refused_with_a_next_attempt() {
        on_large_stack(|| {
            let estimate = reference()["estimate"].as_array().unwrap().clone();
            let base = &estimate[0];
            let n = base["rows"]["outcome"].as_array().unwrap().len();
            let refused = |result: Result<StudyResult, CausalError>, needle: &str| {
                let error = result.err().expect("an estimate was published");
                assert_eq!(error.reason_code(), Some("option_not_applicable"), "{error}");
                assert!(error.to_string().contains(needle), "{error}");
                assert!(error.to_string().contains("Next attempt"), "{error}");
            };
            // A clustered panel without labels: no whole cluster can be formed.
            refused(
                run(builder(table(&base["rows"]))
                    .sampling_design(DeclaredSamplingDesign::ClusteredPanel)),
                "no cluster labels",
            );
            // Labels for fewer rows than the table has.
            refused(
                run(builder(table(&base["rows"]))
                    .sampling_design(DeclaredSamplingDesign::ClusteredPanel)
                    .score_dependence(ScoreDependence::clusters(vec![1_u32; n - 1]))),
                "label is needed for every row",
            );
            // Labels beside a design that does not read them, and a lag beside a clustered panel.
            refused(
                run(builder(table(&base["rows"]))
                    .score_dependence(ScoreDependence::clusters(vec![1_u32; n]))),
                "disagree",
            );
            refused(
                run(builder(table(&base["rows"]))
                    .sampling_design(DeclaredSamplingDesign::DependentTimeSeries)
                    .score_dependence(ScoreDependence::clusters((0..n as u32).collect::<Vec<_>>()))),
                "disagree",
            );
            // Sampling weights declared on an estimator the route does not run.
            let declared =
                antecedent_estimate::SamplingWeights::new(vec![1.0; n]).unwrap();
            refused(
                run(builder(table(&base["rows"])).estimator(
                    antecedent_estimate::LinearAdjustmentAte::new().with_sampling_weights(declared),
                )),
                "sampling_weights.route_unsupported",
            );
            // One cluster carries no between-cluster variation: a typed estimation refusal.
            let error = run(builder(table(&base["rows"]))
                .sampling_design(DeclaredSamplingDesign::ClusteredPanel)
                .score_dependence(ScoreDependence::clusters(vec![3_u32; n])))
            .err()
            .expect("one cluster");
            assert!(error.to_string().contains("at least two clusters"), "{error}");
        });
    }
}

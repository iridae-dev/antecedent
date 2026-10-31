//! Bayesian functional, mediation, counterfactual, derivative and CATE-mixture routes.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::float_cmp,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, IdentifierId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, CounterfactualQuery, DerivativeScale,
    DerivativeWeighting, ExecutionContext, Intervention, InterventionalDistributionQuery,
    MediationContrast, MediationQuery, PathSpecificEffectQuery, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::{GraphPosterior, SamplingSummary, set_edge};
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{Admg, Dag, DenseNodeId};
use antecedent_prob::InferenceDiagnostics;

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(128).prior_scale(1_000.0))
}

fn path_fixture() -> (TabularData, Dag, PathSpecificEffectQuery) {
    let mut t = Vec::new();
    let mut m = Vec::new();
    let mut y = Vec::new();
    for (tv, mv, yv, count) in [
        (0.0, 0.0, 0.0, 40),
        (0.0, 0.0, 1.0, 10),
        (0.0, 1.0, 0.0, 10),
        (0.0, 1.0, 1.0, 40),
        (1.0, 0.0, 0.0, 10),
        (1.0, 0.0, 1.0, 10),
        (1.0, 1.0, 0.0, 10),
        (1.0, 1.0, 1.0, 70),
    ] {
        t.extend(std::iter::repeat_n(tv, count));
        m.extend(std::iter::repeat_n(mv, count));
        y.extend(std::iter::repeat_n(yv, count));
    }
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let query = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
        .with_path_nodes([VariableId::from_raw(1)]);
    (data, dag, query)
}

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
    let mut dag = Dag::with_variables(3);
    for (s, t) in [(0, 1), (0, 2), (1, 2)] {
        dag.insert_directed(DenseNodeId::from_raw(s), DenseNodeId::from_raw(t)).unwrap();
    }
    (data, dag)
}

fn expand_contingency(pin: &serde_json::Value) -> TabularData {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (i, name) in columns.iter().enumerate() {
            values[i].extend(std::iter::repeat_n(cell[*name].as_f64().unwrap(), count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(values.iter()).map(|(n, c)| (*n, c.as_slice())).collect();
    TabularData::from_f64_columns(pairs).unwrap()
}

#[test]
fn functional_bayesian_path_distribution_and_admg() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/functional_validation/expected.json"
    ))
    .unwrap();
    let ctx = ExecutionContext::for_tests(18);
    for accepted in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let (data, dag, query) = path_fixture();
            let builder = if accepted {
                Study::tabular(data.clone()).graph(AcceptedGraph::from(dag))
            } else {
                Study::tabular(data.clone()).graph(dag)
            };
            let result = builder
                .query(CausalQuery::PathSpecific(query))
                .inference(bayes())
                .refute(suite)
                .build()
                .unwrap()
                .prepare(&ctx)
                .unwrap()
                .estimate(&data, &ctx)
                .unwrap();
            assert_eq!(result.logical_plan.estimator.as_deref(), Some("functional.effect"));
            assert!(result.posterior.is_some());
            assert_eq!(
                result.posterior.as_ref().unwrap().diagnostics.backend_id.as_ref(),
                "functional.dirichlet"
            );
            // Closed-form 0.3; the Dirichlet posterior mean sits 0.002 away on this table, so
            // a band of 0.01 still passes honest runs and fails a 0.05 bias.
            assert!(
                (result.estimate().as_effect().unwrap().ate - pin["path_effect"].as_f64().unwrap())
                    .abs()
                    < 0.01,
                "path suite={suite:?} ate={}",
                result.estimate().as_effect().unwrap().ate
            );
            assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));
            let reports = match suite {
                RefuteSuite::None => 0,
                RefuteSuite::Cheap => pin["cheap_path_reports"].as_u64().unwrap(),
                _ => pin["full_path_reports"].as_u64().unwrap(),
            };
            assert_eq!(
                result.refutations.len() as u64,
                reports,
                "path suite={suite:?} must run the path subset-stability suite"
            );
            assert!(result.refutations.iter().all(|r| r.refuter.starts_with("path.")));
            // Subsample refits are centred on the full-sample estimate by construction and
            // judged against single-replicate spread, so they cannot fail: not evidence.
            assert!(
                result.refutations.iter().all(|r| !r.informative),
                "path subset stability must not be reported as informative"
            );
        }
    }

    let admg_pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/admg_frontdoor_functional/expected.json"
    ))
    .unwrap();
    let data = expand_contingency(&admg_pin);
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(2), 0.0, 1.0);

    // Dag + general.id must execute the functional evaluator (not bayesian.gcomp).
    // On the observed chain the ID functional is the g-formula, not the ADMG
    // front-door (0.282); pin against the Frequentist functional on the same pair.
    let _ = include_str!("../../../conformance/estimate/frontdoor/expected.json");
    let _ = include_str!("../../../conformance/identify/general_id_frontdoor/expected.json");
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let freq = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .identifier(IdentifierId::GeneralId)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let rider = Study::tabular(data)
        .graph(dag)
        .query(query)
        .identifier(IdentifierId::GeneralId)
        .inference(bayes())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert_eq!(rider.logical_plan.estimator.as_deref(), Some("functional.effect"));
    assert_ne!(rider.logical_plan.estimator.as_deref(), Some("bayesian.gcomp"));
    assert!(rider.posterior.is_some());
    // Measured gap between the Dirichlet posterior mean and the frequentist functional is 0.0012.
    assert!(
        (rider.estimate().as_effect().unwrap().ate - freq.estimate().as_effect().unwrap().ate)
            .abs()
            < 0.01
    );
}

/// Bayesian front-door ADMG ATE on every licensed coordinate: explicit and
/// accepted structure × `none`/`cheap`/`full`, fresh and prepared. The frozen
/// binary law has front-door effect 0.282 (`admg_frontdoor_functional`); the
/// Dirichlet posterior mean must sit within 0.02 of it, and the 90% credible
/// interval must contain it.
#[test]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the rounded quantile position (len-1)*q with q in [0, 1] is a non-negative index below len"
)]
fn admg_frontdoor_bayesian_all_structures_and_validation() {
    let admg_pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/admg_frontdoor_functional/expected.json"
    ))
    .unwrap();
    let truth = admg_pin["frequentist"]["expected_ate"].as_f64().unwrap();
    let data = expand_contingency(&admg_pin);
    let mut admg = Admg::with_variables(3);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    admg.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    admg.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(2), 0.0, 1.0);
    let ctx = ExecutionContext::for_tests(18);
    for accepted in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let builder = if accepted {
                Study::tabular(data.clone()).graph(AcceptedGraph::from(admg.clone()))
            } else {
                Study::tabular(data.clone()).graph(admg.clone())
            };
            let study =
                builder.query(query.clone()).inference(bayes()).refute(suite).build().unwrap();
            let fresh = study.clone().run(&ctx).unwrap();
            let click = study.prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap();
            assert!(
                click.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
                "prepared ADMG click must reuse identification"
            );
            for result in [&fresh, &click] {
                let label = format!("accepted={accepted} suite={suite:?}");
                assert_eq!(result.support_status().unwrap().as_str(), "licensed", "{label}");
                assert_eq!(result.logical_plan.identifier.as_deref(), Some("general.id"));
                assert_eq!(result.logical_plan.estimator.as_deref(), Some("functional.effect"));
                let posterior = result.posterior.as_ref().expect("Bayesian ADMG posterior");
                assert!(
                    (result.estimate().as_effect().unwrap().ate - truth).abs() < 0.02,
                    "{label}: posterior mean {} vs front-door truth {truth}",
                    result.estimate().as_effect().unwrap().ate
                );
                let col = posterior.effect_column().expect("effect draws");
                let mut draws = posterior.draws.column(col).unwrap().to_vec();
                draws.sort_by(f64::total_cmp);
                let at = |q: f64| draws[((draws.len() - 1) as f64 * q).round() as usize];
                assert!(at(0.05) <= truth && truth <= at(0.95), "{label}: 90% interval");
                match suite {
                    RefuteSuite::None => {
                        assert!(result.refutations.is_empty(), "{label}: none runs no refuter");
                    }
                    _ => assert!(!result.refutations.is_empty(), "{label}: refuters must run"),
                }
            }
            assert!(
                (fresh.estimate().as_effect().unwrap().ate
                    - click.estimate().as_effect().unwrap().ate)
                    .abs()
                    < 1e-12
            );
        }
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one scenario kept in one body: the fixture, the run and the pinned assertions for distribution bayesian known truth"
)]
fn distribution_bayesian_known_truth() {
    let _ = include_str!("../../../conformance/estimate/interventional_distribution/expected.json");
    let _ = include_str!("../../../conformance/context/path_specific_natural/expected.json");
    let _ = include_str!("../../../conformance/identify/general_id_frontdoor/expected.json");
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/functional_validation/expected.json"
    ))
    .unwrap();
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
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    let query = InterventionalDistributionQuery::new(
        VariableId::from_raw(1),
        [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
    );
    let ctx = ExecutionContext::for_tests(18);
    let result = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::Distribution(query))
        .inference(bayes())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap()
        .estimate(&data, &ctx)
        .unwrap();
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("functional.distribution"));
    let dist = result.distribution().unwrap();
    // Measured deviation from the closed-form mean is 0.003; 0.01 still fails a 0.05 bias.
    assert!((dist.mean - pin["distribution_mean"].as_f64().unwrap()).abs() < 0.01);
    let mass: f64 = dist.atoms.iter().map(|a| a.probability).sum();
    // Posterior-mean atom probabilities of a Dirichlet draw set sum to one to rounding.
    assert!((mass - 1.0).abs() < 1e-9, "bayesian atoms must be a coherent law, mass={mass}");
    let posterior = result.posterior.as_ref().unwrap();
    let atom_mean: f64 =
        dist.atoms.iter().map(|a| a.probability * a.outcomes[0].1.as_f64().unwrap()).sum();
    assert!((dist.mean - atom_mean).abs() < 1e-12);
    assert_eq!(posterior.draws.n_quantities(), dist.atoms.len() + 1);
    for draw in 0..posterior.draws.n_draws {
        let p0 = posterior.draws.column(1).unwrap()[draw];
        let p1 = posterior.draws.column(2).unwrap()[draw];
        assert!((p0 + p1 - 1.0).abs() < 1e-12);
        assert!((p1 - posterior.draws.column(0).unwrap()[draw]).abs() < 1e-12);
    }

    // A distribution need not have a scalar mean: the same licensed path
    // must produce joint probability draws for joint outcomes and IDC tables.
    for conditional in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let mut dag = Dag::with_variables(3);
            dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
            dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
            dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
            let query = InterventionalDistributionQuery::new(
                VariableId::from_raw(1),
                [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
            );
            let query = if conditional {
                query.with_conditioning(Arc::from([VariableId::from_raw(2)]))
            } else {
                query.with_outcomes(Arc::from([VariableId::from_raw(1), VariableId::from_raw(2)]))
            };
            let result = Study::tabular(data.clone())
                .graph(dag)
                .query(CausalQuery::Distribution(query))
                .inference(bayes())
                .refute(suite)
                .build()
                .unwrap()
                .prepare(&ctx)
                .unwrap()
                .estimate(&data, &ctx)
                .unwrap();
            let dist = result.distribution().unwrap();
            let posterior = result.posterior.as_ref().unwrap();
            assert!(dist.mean.is_nan());
            assert!(posterior.effect_column().is_none());
            assert_eq!(posterior.draws.n_quantities(), dist.atoms.len());
            for (i, atom) in dist.atoms.iter().enumerate() {
                assert_eq!(atom.probability, posterior.summaries.mean[i]);
                let group: Vec<_> = dist
                    .atoms
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| a.conditioning == atom.conditioning)
                    .map(|(j, _)| j)
                    .collect();
                for draw in 0..posterior.draws.n_draws {
                    let mass: f64 =
                        group.iter().map(|&j| posterior.draws.column(j).unwrap()[draw]).sum();
                    assert!((mass - 1.0).abs() < 1e-12);
                }
            }
            if suite != RefuteSuite::None {
                assert!(
                    result
                        .refutations
                        .iter()
                        .any(|r| r.refuter.as_ref() == "distribution.normalization")
                );
            }
        }
    }
}

#[test]
fn mediation_and_counterfactual_bayesian_pins() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/staged_static_kinds/expected.json"
    ))
    .unwrap();
    let (data, dag) = mediation_scm();
    let ctx = ExecutionContext::for_tests(13);
    let control = pin["control"].as_f64().unwrap();
    let active = pin["active"].as_f64().unwrap();
    for (key, contrast) in [
        ("direct", MediationContrast::NaturalDirect),
        ("indirect", MediationContrast::NaturalIndirect),
    ] {
        let mut query = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            Arc::from([VariableId::from_raw(1)]),
            contrast,
        );
        query.control = Intervention::set(query.treatment, Value::f64(control));
        query.active = Intervention::set(query.treatment, Value::f64(active));
        let result = Study::tabular(data.clone())
            .graph(dag.clone())
            .query(CausalQuery::Mediation(query))
            .inference(bayes())
            .refute(RefuteSuite::Cheap)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap()
            .estimate(&data, &ctx)
            .unwrap();
        assert!(result.posterior.is_some());
        // Closed-form effects (1.8 direct, 4.8 indirect); the posterior mean is within 0.007.
        assert!(
            (result.estimate().as_effect().unwrap().ate - pin[key].as_f64().unwrap()).abs() < 0.03,
            "{key} {}",
            result.estimate().as_effect().unwrap().ate
        );
        assert_eq!(result.logical_plan.estimator.as_deref(), Some("mediation.linear"));
    }

    let query = CounterfactualQuery::new(
        VariableId::from_raw(2),
        Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(active))]),
    )
    .with_control_level(control);
    let result = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(CausalQuery::Counterfactual(query.clone()))
        .inference(bayes())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap()
        .estimate(&data, &ctx)
        .unwrap();
    let cf = result.counterfactual().unwrap();
    assert!(result.posterior.is_some());
    // Closed-form mean ITE 6.6; the posterior mean is 0.03 away, so 0.08 replaces the old 0.25.
    assert!((cf.mean_ite - pin["counterfactual_mean"].as_f64().unwrap()).abs() < 0.08);
    assert_eq!(cf.unit_effects.len(), 500);
    let unit_mean = cf.unit_effects.iter().sum::<f64>() / cf.unit_effects.len() as f64;
    assert!(
        (unit_mean - cf.mean_ite).abs() < 0.05,
        "unit posterior mean {unit_mean} must match mean ITE {}",
        cf.mean_ite
    );
    let freq = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::Counterfactual(query))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap()
        .estimate(&data, &ctx)
        .unwrap();
    let point = freq.counterfactual().unwrap();
    let l1 = cf
        .unit_effects
        .iter()
        .zip(point.unit_effects.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / 500.0;
    assert!(l1 > 1e-6, "unit effects must be the per-draw AAP posterior, not point AAP");
    assert!(l1 < 1.0, "unit posterior should stay on the same SCM, l1={l1}");
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("gcm.fit"));
}

#[test]
fn accepted_counterfactual_matches_explicit_bayesian() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/staged_static_kinds/expected.json"
    ))
    .unwrap();
    let (data, dag) = mediation_scm();
    let ctx = ExecutionContext::for_tests(13);
    let query = CounterfactualQuery::new(
        VariableId::from_raw(2),
        Arc::from([Intervention::set(
            VariableId::from_raw(0),
            Value::f64(pin["active"].as_f64().unwrap()),
        )]),
    )
    .with_control_level(pin["control"].as_f64().unwrap());
    let run = |graph: antecedent::AcceptedGraph| {
        Study::tabular(data.clone())
            .graph(graph)
            .query(CausalQuery::Counterfactual(query.clone()))
            .inference(bayes())
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap()
            .estimate(&data, &ctx)
            .unwrap()
    };
    let explicit = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(CausalQuery::Counterfactual(query.clone()))
        .inference(bayes())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap()
        .estimate(&data, &ctx)
        .unwrap();
    let accepted = run(AcceptedGraph::from(dag));
    assert_eq!(accepted.structure_source, antecedent::StructureSource::Accepted);
    assert_eq!(explicit.structure_source, antecedent::StructureSource::Explicit);
    let explicit_cf = explicit.counterfactual().unwrap();
    let accepted_cf = accepted.counterfactual().unwrap();
    assert!((accepted_cf.mean_ite - explicit_cf.mean_ite).abs() < 1e-12);
    assert_eq!(accepted_cf.unit_effects, explicit_cf.unit_effects);
    assert_eq!(accepted.logical_plan.estimator.as_deref(), Some("gcm.fit"));
}

/// Known-truth pins for all six Bayesian derivative functionals on explicit and
/// accepted DAGs (`conformance/response/staged_derivatives`): each asserts the
/// point value and that the published credible interval brackets the truth.
#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one scenario kept in one body: the fixture, the run and the pinned assertions for derivative bayesian pins"
)]
fn derivative_bayesian_pins() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/response/staged_derivatives/expected.json"
    ))
    .unwrap();
    let tolerance = pin["tolerance"].as_f64().unwrap();
    let a: Vec<_> = (0..400)
        .map(|i| 2.0 + (f64::from(i) * 0.71).sin() + 0.2 * (f64::from(i) * 0.13).cos())
        .collect();
    let b: Vec<_> = (0..400).map(|i| (f64::from(i) * 1.13).cos()).collect();
    let y: Vec<_> = a.iter().zip(&b).map(|(a, b)| 5.0 + 2.0 * a - 0.5 * b).collect();
    let v: Vec<_> = a.iter().zip(&b).map(|(a, b)| 1.0 + 0.25 * a + 1.5 * b).collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("y", y.as_slice()),
        ("v", v.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(0, 2), (0, 3), (1, 2), (1, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let ids =
        |xs: &[u32]| -> Arc<[VariableId]> { xs.iter().map(|&x| VariableId::from_raw(x)).collect() };
    let point = |scale| ResponseFunctional::PointDerivative {
        outcome: VariableId::from_raw(2),
        treatment: VariableId::from_raw(0),
        at: 2.0,
        order: 1,
        scale,
    };
    let cases = [
        ("point", point(DerivativeScale::Identity)),
        ("elasticity", point(DerivativeScale::LogLog)),
        ("semi_treatment", point(DerivativeScale::LogTreatment)),
        ("semi_outcome", point(DerivativeScale::LogOutcome)),
        (
            "average",
            ResponseFunctional::AverageDerivative {
                outcome: VariableId::from_raw(2),
                treatment: VariableId::from_raw(0),
                weighting: DerivativeWeighting::Observed,
            },
        ),
        (
            "jacobian",
            ResponseFunctional::Jacobian {
                outcomes: ids(&[2, 3]),
                treatments: ids(&[0, 1]),
                at: Arc::from([2.0, 0.0]),
                scale: DerivativeScale::Identity,
            },
        ),
        (
            "directional",
            ResponseFunctional::DirectionalDerivative {
                outcomes: ids(&[2, 3]),
                treatments: ids(&[0, 1]),
                at: Arc::from([2.0, 0.0]),
                direction: Arc::from([1.0, 2.0]),
            },
        ),
    ];
    let ctx = ExecutionContext::for_tests(18);
    for (key, functional) in cases {
        let truth: Vec<f64> = pin[key].as_f64().map_or_else(
            || pin[key].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect(),
            |x| vec![x],
        );
        for accepted in [false, true] {
            let builder = if accepted {
                Study::tabular(data.clone()).graph(AcceptedGraph::from(graph.clone()))
            } else {
                Study::tabular(data.clone()).graph(graph.clone())
            };
            let result = builder
                .query(CausalQuery::Response(ResponseQuery::new(functional.clone())))
                .response_options(ContinuousResponseOptions {
                    bandwidth: Some(0.35),
                    ..Default::default()
                })
                // The derivative route uses Rubin's Bayesian bootstrap; this is the
                // compatibility declaration, not a Laplace coefficient posterior.
                .inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(64)))
                .refute(RefuteSuite::None)
                .build()
                .unwrap()
                .prepare(&ctx)
                .unwrap()
                .estimate(&data, &ctx)
                .unwrap();
            let label = format!("{key} accepted={accepted}");
            assert_ne!(result.logical_plan.estimator.as_deref(), Some("bayesian.gcomp"), "{label}");
            assert_ne!(
                result.logical_plan.estimator.as_deref(),
                Some("response.bayesian"),
                "{label}"
            );
            let response = result.response().unwrap();
            let ResponseIdentification::PointIdentified(value) = &response.estimate else {
                panic!("{label}: unidentified")
            };
            let got = match value {
                ResponseValue::Scalar(x) => vec![*x],
                ResponseValue::Jacobian { values, .. } | ResponseValue::Vector(values) => {
                    values.to_vec()
                }
                other => panic!("{label}: unexpected {other:?}"),
            };
            let (lower, upper): (Vec<f64>, Vec<f64>) = match &response.uncertainty {
                ResponseUncertainty::Scalar { lower, upper, .. } => (vec![*lower], vec![*upper]),
                ResponseUncertainty::PointwiseBand { lower, upper, .. } => {
                    (lower.to_vec(), upper.to_vec())
                }
                other => {
                    panic!("{label}: Bayesian derivative must publish an interval, got {other:?}")
                }
            };
            assert_eq!(got.len(), truth.len(), "{label}");
            for j in 0..truth.len() {
                // Measured deviations on this noise-free law: Jacobian and directional values
                // within 1e-8, point-type scales within 0.003, the observed-weight average
                // within 0.05. The shared fixture band (0.15) is wider than any of them.
                let key_tolerance = match key {
                    "jacobian" | "directional" => 1e-6,
                    "average" => 0.1,
                    _ => 0.02,
                };
                assert!(
                    (got[j] - truth[j]).abs() < tolerance.min(key_tolerance),
                    "{label}[{j}]: {} != {}",
                    got[j],
                    truth[j]
                );
                // The fixture is noise-free, so a credible interval may be a
                // numerical sliver; allow one ulp-scale slack around the truth.
                let slack = 1e-9 * truth[j].abs().max(1.0);
                assert!(
                    lower[j] - slack <= truth[j] && truth[j] <= upper[j] + slack,
                    "{label}[{j}]: credible interval [{}, {}] misses truth {}",
                    lower[j],
                    upper[j],
                    truth[j]
                );
            }
        }
    }
}

fn mixture_gp() -> GraphPosterior {
    let weights = [0.5, 0.3, 0.2];
    let direct = set_edge(0, 3, 0, 1, true);
    let adjusted = set_edge(set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
    let unidentified = set_edge(0, 3, 1, 0, true);
    let mut marginals = vec![0.0; 9];
    marginals[1] = weights[0] + weights[1];
    marginals[3] = weights[2];
    marginals[6] = weights[1];
    marginals[7] = weights[1];
    GraphPosterior::new(
        3,
        weights.to_vec(),
        vec![direct, adjusted, unidentified],
        marginals.clone(),
        marginals,
        SamplingSummary {
            ess: 1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
            diagnostics: InferenceDiagnostics::analytic("known_truth_mixtures"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
}

/// Treatment, outcome and confounder columns of the known-truth mixture: `y = slope * t + 2 z`
/// plus a deterministic symmetric disturbance, with `z` and `t` positively associated.
fn mixture_columns(n: usize, slope: f64) -> [Vec<f64>; 3] {
    let mut treatment = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    let mut confounder = Vec::with_capacity(n);
    for _ in 0..(n / 16) {
        for (z, t, count) in [(0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)] {
            for row in 0..count {
                let epsilon = if row % 2 == 0 { -0.2 } else { 0.2 };
                treatment.push(t);
                confounder.push(z);
                outcome.push(slope * t + 2.0 * z + epsilon);
            }
        }
    }
    [treatment, outcome, confounder]
}

fn mixture_table(columns: &[Vec<f64>; 3]) -> TabularData {
    TabularData::from_f64_columns([
        ("t", columns[0].as_slice()),
        ("y", columns[1].as_slice()),
        ("z", columns[2].as_slice()),
    ])
    .unwrap()
}

fn mixture_data(n: usize) -> TabularData {
    mixture_table(&mixture_columns(n, 2.0))
}

#[test]
fn conditional_graph_posterior_retains_unidentified_mass() {
    let _ = include_str!("../../../conformance/bayesian/known_truth_mixtures/expected.json");
    let data = mixture_data(160);
    let inner = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_effect_modifiers([VariableId::from_raw(2)]);
    let query = ConditionalEffectQuery::try_new(inner).unwrap();
    let ctx = ExecutionContext::for_tests(18);
    let freq = Study::tabular(data.clone())
        .graph_posterior(mixture_gp())
        .query(CausalQuery::ConditionalEffect(query.clone()))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::Cheap)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let bayes = Study::tabular(data)
        .graph_posterior(mixture_gp())
        .query(CausalQuery::ConditionalEffect(query))
        .inference(bayes())
        .refute(RefuteSuite::Cheap)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert_eq!(freq.identification.status, antecedent_core::IdentificationStatus::GraphDependent);
    assert!(
        freq.diagnostics.iter().any(|d| d.message.contains("unidentified_mass=0.2")),
        "Frequentist mixture must publish the unidentified mass"
    );
    assert!(
        bayes.posterior.as_ref().unwrap().assumptions.entries.iter().any(|record| {
            matches!(&record.assumption, antecedent_core::Assumption::PriorRestriction(_))
        }),
        "graph mixtures must retain component priors"
    );
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    let unidentified = pin["static_average_effect"]["expected_unidentified_mass"].as_f64().unwrap();
    assert!(
        bayes.posterior.as_ref().is_some_and(|p| (p.unidentified_mass - unidentified).abs() < 1e-9),
        "bayes mass {:?}",
        bayes.posterior.as_ref().map(|p| p.unidentified_mass)
    );
    assert!(
        bayes.posterior.as_ref().is_some_and(|p| p.assumptions.entries.iter().any(|a| {
            matches!(
                &a.assumption,
                antecedent_core::Assumption::ParametricRestriction(r)
                    if r.id.as_ref() == "envelope.conditional_on_identification"
            )
        })),
        "Bayesian envelope must record that E[τ] is conditional on identification"
    );
    // Distinct adjustment sets are GraphDependentAtoms: withhold the scalar,
    // keep unidentified mass, publish the identified set. Both identified CATE
    // atoms recover the structural slope 2 on Y = 2T + 2Z.
    assert!(
        bayes.estimate().as_effect().unwrap().ate.is_nan(),
        "disagreeing CATE atoms must withhold estimate.ate, got {}",
        bayes.estimate().as_effect().unwrap().ate
    );
    assert!(
        bayes.diagnostics.iter().any(|d| {
            d.code.as_ref() == "estimate.graph_posterior.structural_aggregation"
                && d.message.contains("graph_dependent_atoms")
        }),
        "Bayesian CATE mixture must disclose GraphDependentAtoms"
    );
    let bayes_post = bayes.posterior.as_ref().unwrap();
    assert_ne!(
        bayes_post.diagnostics.backend_id.as_ref(),
        freq.posterior.as_ref().map_or("", |p| p.diagnostics.backend_id.as_ref()),
        "Bayesian graph-posterior CATE must not be a silent Frequentist reuse"
    );
    assert_ne!(bayes.logical_plan.estimator.as_deref(), freq.logical_plan.estimator.as_deref());
}

#[test]
fn transfer_without_mapping_fails_closed() {
    let (data, dag) = mediation_scm();
    let ctx = ExecutionContext::for_tests(1);
    let junk = Arc::<[u8]>::from(vec![1_u8, 2, 3]);
    let cfg = BayesianConfig::conjugate().n_draws(8).prior_from_artifact(junk.to_vec(), None);
    let mut query = MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        Arc::from([VariableId::from_raw(1)]),
        MediationContrast::NaturalDirect,
    );
    query.control = Intervention::set(query.treatment, Value::f64(0.2));
    query.active = Intervention::set(query.treatment, Value::f64(0.8));
    let err = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(CausalQuery::Mediation(query))
        .inference(InferenceMode::Bayesian(cfg.clone()))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap_err();
    assert!(err.to_string().contains("require an explicit coefficient mapping"), "{err}");

    let cf = CounterfactualQuery::new(
        VariableId::from_raw(2),
        Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(0.8))]),
    )
    .with_control_level(0.2);
    let err = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(CausalQuery::Counterfactual(cf.clone()))
        .inference(InferenceMode::Bayesian(cfg))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap_err();
    assert!(err.to_string().contains("declared mechanism mapping"), "{err}");

    let mapped = BayesianConfig::conjugate().n_draws(8).prior_from_artifact(
        junk.to_vec(),
        Some(antecedent_io::PriorMapping::IdenticalCoefficientSubspace),
    );
    let err = Study::tabular(data)
        .graph(dag)
        .query(CausalQuery::Counterfactual(cf))
        .inference(InferenceMode::Bayesian(mapped))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap_err();
    assert!(err.to_string().contains("declared mechanism mapping"), "{err}");
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one scenario kept in one body: the fixture, the run and the pinned assertions for static mediation named prior hydrates only the outcome treatment slope"
)]
fn static_mediation_named_prior_hydrates_only_the_outcome_treatment_slope() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/static_mediation_prior_transfer/expected.json"
    ))
    .unwrap();
    let kinds: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/staged_static_kinds/expected.json"
    ))
    .unwrap();
    let (data, dag) = mediation_scm();
    let ctx = ExecutionContext::for_tests(pin["seed"].as_u64().unwrap());
    let control = kinds["control"].as_f64().unwrap();
    let active = kinds["active"].as_f64().unwrap();
    let ate = AverageEffectQuery::with_levels(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        control,
        active,
    );
    let source = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(ate)
        .inference(bayes())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let bytes =
        antecedent::io::encode_causal_posterior_bytes(source.posterior.as_ref().unwrap(), "source")
            .unwrap();
    let mut query = MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        Arc::from([VariableId::from_raw(1)]),
        MediationContrast::NaturalDirect,
    );
    query.control = Intervention::set(query.treatment, Value::f64(control));
    query.active = Intervention::set(query.treatment, Value::f64(active));
    let mapped = BayesianConfig::conjugate().n_draws(64).prior_from_artifact(
        bytes,
        Some(antecedent_io::PriorMapping::EffectFunctional { source_quantity: "ate".into() }),
    );
    let transferred = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(CausalQuery::Mediation(query.clone()))
        .inference(InferenceMode::Bayesian(mapped))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!(transferred.posterior.is_some());
    let assumptions = format!("{:?}", transferred.posterior.as_ref().unwrap().assumptions);
    assert!(
        assumptions.contains("mapped ATE/Δ prior hydrated onto outcome-mechanism"),
        "EffectFunctional must bind only the outcome-mechanism NDE slope: {assumptions}"
    );
    assert!(
        assumptions.contains("[y]"),
        "only the outcome mechanism may receive the mapped ATE/Δ prior: {assumptions}"
    );
    assert!(
        !assumptions.contains("[y, m]") && !assumptions.contains("[m, y]"),
        "mediator mechanism must keep isotropic prior_scale: {assumptions}"
    );
    let source_ate = source.estimate().as_effect().unwrap().ate;
    assert!(
        assumptions.contains("implied NDE/ATE mean"),
        "Δ-scaled slope prior must record implied NDE = source ATE ({source_ate}); assumptions={assumptions}"
    );
    assert!(
        (source_ate - kinds["total"].as_f64().unwrap()).abs() < 0.25,
        "source ATE {source_ate} must be the Δ-scaled NDE prior location"
    );
    assert_eq!(pin["compatibility_filter"].as_str().unwrap(), "outcome-mechanism ATE/Δ hydrate");
    let mapping = antecedent_estimate::HydrateMapping::NamedParameters {
        pairs: vec![("slope_t".into(), "coef_a".into())],
    };
    let quantities = [
        antecedent_prob::PosteriorQuantityKind::Scalar { name: Arc::from("slope_t") },
        antecedent_prob::PosteriorQuantityKind::ResidualVariance,
    ];
    let bridge = antecedent_estimate::MediationPriorBridge {
        mapping: &mapping,
        quantities: &quantities,
        mean: &[3.0, 1.0],
        sd: &[0.2, 0.1],
        source_contrast: None,
    };
    let estimator =
        antecedent_estimate::BayesianGComputationAte { n_draws: 64, ..Default::default() };
    let (_, posterior) = antecedent_estimate::estimate_static_mediation_bayesian_outcome_prior(
        &antecedent_estimate::StaticMediationProblem {
            data: &data,
            graph: &dag,
            query: &query,
            extra: &[],
        },
        antecedent_core::AssumptionSet::default(),
        &estimator,
        source.identification.status,
        Some(bridge),
        &ctx,
    )
    .unwrap();
    assert!(
        posterior.assumptions.entries.iter().any(|record| {
            matches!(&record.assumption, antecedent_core::Assumption::PriorRestriction(prior)
            if prior.id.as_ref() == "external_named_prior")
                && matches!(&record.scope, antecedent_core::AssumptionScope::Variables { variables }
                if variables.as_ref() == [VariableId::from_raw(2)])
        }),
        "the named treatment-slope prior must reach the outcome mechanism"
    );
    assert!(
        !posterior.assumptions.entries.iter().any(|record| {
            matches!(&record.assumption, antecedent_core::Assumption::PriorRestriction(prior)
            if prior.id.as_ref() == "external_named_prior")
                && matches!(&record.scope, antecedent_core::AssumptionScope::Variables { variables }
                if variables.as_ref() == [VariableId::from_raw(1)])
        }),
        "the shared raw coefficient name must not hydrate the mediator mechanism"
    );
    assert!(!format!("{:?}", posterior.assumptions).contains("mapped ATE/Δ"));
    let bad_mapping = antecedent_estimate::HydrateMapping::NamedParameters {
        pairs: vec![("slope_t".into(), "typo".into())],
    };
    let err = antecedent_estimate::estimate_static_mediation_bayesian_outcome_prior(
        &antecedent_estimate::StaticMediationProblem {
            data: &data,
            graph: &dag,
            query: &query,
            extra: &[],
        },
        antecedent_core::AssumptionSet::default(),
        &estimator,
        source.identification.status,
        Some(antecedent_estimate::MediationPriorBridge { mapping: &bad_mapping, ..bridge }),
        &ctx,
    )
    .unwrap_err();
    assert!(err.to_string().contains("typo"));
}

#[test]
fn bayesian_does_not_rewrite_functional_estimators() {
    let (data, dag, query) = path_fixture();
    let study = Study::tabular(data)
        .graph(dag)
        .query(CausalQuery::PathSpecific(query))
        .inference(bayes())
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let plan = study.compile(&ExecutionContext::for_tests(1)).unwrap();
    assert_eq!(plan.logical.record.estimator.as_deref(), Some("functional.effect"));
    assert_ne!(plan.logical.record.estimator.as_deref(), Some("bayesian.gcomp"));
    for estimator_first in [false, true] {
        let (data, dag, query) = path_fixture();
        let builder = Study::tabular(data).graph(dag).inference(bayes()).refute(RefuteSuite::None);
        let builder = if estimator_first {
            builder
                .estimator(antecedent::EstimatorId::BayesianGcomp)
                .query(CausalQuery::PathSpecific(query))
        } else {
            builder
                .query(CausalQuery::PathSpecific(query))
                .estimator(antecedent::EstimatorId::BayesianGcomp)
        };
        let study = builder.build().unwrap();
        assert!(
            study.compile(&ExecutionContext::for_tests(1)).is_err(),
            "an explicit incompatible estimator must not be silently replaced"
        );
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one scenario kept in one body: the fixture, the run and the pinned assertions for static effect and response prior transfer"
)]
fn static_effect_and_response_prior_transfer() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/static_effect_prior_transfer/expected.json"
    ))
    .unwrap();
    let _ =
        include_str!("../../../conformance/bayesian/static_response_prior_transfer/expected.json");
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for i in 0..n {
        let zi = if i % 2 == 0 { 0.0 } else { 1.0 };
        let ti = if i % 3 == 0 { 1.0 } else { 0.0 };
        t.push(ti);
        z.push(zi);
        y.push(2.0 * ti + zi);
    }
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let ctx = ExecutionContext::for_tests(pin["seed"].as_u64().unwrap());
    let source = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .inference(bayes())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let bytes =
        antecedent::io::encode_causal_posterior_bytes(source.posterior.as_ref().unwrap(), "source")
            .unwrap();
    // The target prior design must include both intervention coordinates.
    let joint_query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([
            Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
            Intervention::set(VariableId::from_raw(2), Value::f64(1.0)),
        ]),
    });
    let joint = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(CausalQuery::Response(joint_query))
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(64).prior_from_artifact(
                bytes.clone(),
                Some(antecedent_io::PriorMapping::IdenticalCoefficientSubspace),
            ),
        ))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!((joint.estimate().as_effect().unwrap().ate - 3.0).abs() < 0.1);
    let catalog =
        antecedent_io::PriorCatalog::from_sources(vec![antecedent_io::PriorSourceRef::with_bytes(
            antecedent_io::PriorSourceMeta::new(
                "source",
                antecedent_io::EstimandFingerprint::new("ate", "t", "y"),
                "NonparametricallyIdentified",
            )
            .with_design(vec![
                antecedent_io::DesignVariableSummary::new(
                    "t",
                    antecedent_io::DesignVariableRole::Treatment,
                ),
                antecedent_io::DesignVariableSummary::new(
                    "y",
                    antecedent_io::DesignVariableRole::Outcome,
                ),
                antecedent_io::DesignVariableSummary::new(
                    "z",
                    antecedent_io::DesignVariableRole::Covariate,
                ),
            ]),
            bytes.clone(),
        )]);
    let ate_target = antecedent_io::TargetDesign::new(
        antecedent_io::EstimandFingerprint::new("ate", "t", "y"),
        ["t", "y", "z"],
    );
    let reports = catalog.filter_compatible(&ate_target);
    assert!(
        reports[0].is_usable(),
        "same-design ATE must pass filter_compatible: {:?}",
        reports[0]
    );
    catalog.require_usable(&ate_target).expect("same-design ATE catalog");
    assert_eq!(pin["compatibility_filter"].as_str().unwrap(), "PriorCatalog.filter_compatible");
    let mapped = BayesianConfig::conjugate().n_draws(128).prior_scale(10.0).prior_from_artifact(
        bytes.clone(),
        Some(antecedent_io::PriorMapping::IdenticalCoefficientSubspace),
    );
    let prepared = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .inference(InferenceMode::Bayesian(mapped))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let click = prepared.estimate(&data, &ctx).unwrap();
    assert!(
        (click.estimate().as_effect().unwrap().ate - pin["true_effect"].as_f64().unwrap()).abs()
            < pin["atol"].as_f64().unwrap()
    );

    let cate = ConditionalEffectQuery::try_new(
        query.clone().with_effect_modifiers([VariableId::from_raw(2)]),
    )
    .unwrap();
    let cate_target = antecedent_io::TargetDesign::new(
        antecedent_io::EstimandFingerprint::new("cate", "t", "y"),
        ["t", "y", "z"],
    );
    let cate_catalog =
        antecedent_io::PriorCatalog::from_sources(vec![antecedent_io::PriorSourceRef::with_bytes(
            antecedent_io::PriorSourceMeta::new(
                "source",
                antecedent_io::EstimandFingerprint::new("ate", "t", "y"),
                "NonparametricallyIdentified",
            )
            .with_mapping(antecedent_io::PriorMapping::EffectFunctional {
                source_quantity: "ate".into(),
            })
            .with_design(vec![
                antecedent_io::DesignVariableSummary::new(
                    "t",
                    antecedent_io::DesignVariableRole::Treatment,
                ),
                antecedent_io::DesignVariableSummary::new(
                    "y",
                    antecedent_io::DesignVariableRole::Outcome,
                ),
            ]),
            bytes.clone(),
        )]);
    assert!(
        cate_catalog.filter_compatible(&cate_target)[0].is_usable(),
        "mapped CATE must pass filter_compatible"
    );
    let cate_cfg = BayesianConfig::conjugate().n_draws(64).prior_from_artifact(
        bytes.clone(),
        Some(antecedent_io::PriorMapping::EffectFunctional { source_quantity: "ate".into() }),
    );
    let cate_result = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(CausalQuery::ConditionalEffect(cate))
        .inference(InferenceMode::Bayesian(cate_cfg))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!(cate_result.posterior.is_some());
    assert!(
        (cate_result.estimate().as_effect().unwrap().ate - pin["true_effect"].as_f64().unwrap())
            .abs()
            < pin["atol"].as_f64().unwrap()
    );

    let wrong =
        antecedent_io::PriorCatalog::from_sources(vec![antecedent_io::PriorSourceRef::with_bytes(
            antecedent_io::PriorSourceMeta::new(
                "wrong",
                antecedent_io::EstimandFingerprint::new("ate", "t", "other"),
                "NonparametricallyIdentified",
            ),
            bytes.clone(),
        )]);
    let rejected = wrong.filter_compatible(&ate_target);
    assert!(
        matches!(
            &rejected[0],
            antecedent_io::CompatibilityReport::Rejected {
                reason: antecedent_io::CompatibilityRejectReason::EstimandMismatch { .. },
                ..
            }
        ),
        "incompatible catalog must be estimand_mismatch: {:?}",
        rejected[0]
    );
    assert_eq!(pin["target_cells"]["incompatible_catalog"].as_str().unwrap(), "estimand_mismatch");

    let curve = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: antecedent_core::ContinuousDomain::new(
            VariableId::from_raw(0),
            antecedent_core::GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    });
    let missing = BayesianConfig::conjugate().n_draws(8).prior_from_artifact(bytes, None);
    let err = Study::tabular(data)
        .graph(dag)
        .query(CausalQuery::Response(curve))
        .inference(InferenceMode::Bayesian(missing))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap_err();
    assert!(err.to_string().contains("response-specific mapping"), "{err}");
}

#[test]
fn class_envelope_prior_transfer_is_per_completion() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/cpdag_ate_envelope/expected.json"
    ))
    .unwrap();
    let _ =
        include_str!("../../../conformance/bayesian/static_effect_prior_transfer/expected.json");
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (i, name) in columns.iter().enumerate() {
            values[i].extend(std::iter::repeat_n(cell[*name].as_f64().unwrap(), count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(values.iter()).map(|(n, c)| (*n, c.as_slice())).collect();
    let data = TabularData::from_f64_columns(pairs).unwrap();
    let mut cpdag = antecedent_graph::Cpdag::with_variables(3);
    cpdag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    cpdag.insert_undirected(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    let ctx = ExecutionContext::for_tests(1);
    let baseline = Study::tabular(data.clone())
        .graph(cpdag.clone())
        .query(query.clone())
        .inference(bayes())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let status = format!("{:?}", baseline.identification.status);
    let mass = baseline.posterior.as_ref().map(|p| p.unidentified_mass);
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let source = Study::tabular(data.clone())
        .graph(dag)
        .query(query.clone())
        .inference(bayes())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let bytes = antecedent::io::encode_causal_posterior_bytes(
        source.posterior.as_ref().unwrap(),
        "cpdag-source",
    )
    .unwrap();
    let mapped = BayesianConfig::conjugate().n_draws(64).prior_from_artifact(
        bytes,
        Some(antecedent_io::PriorMapping::IdenticalCoefficientSubspace),
    );
    let transferred = Study::tabular(data)
        .graph(cpdag)
        .query(query)
        .inference(InferenceMode::Bayesian(mapped))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx);
    assert_eq!(status, "PartiallyIdentified");
    assert_eq!(mass, Some(0.0));
    // The source posterior has three coefficients (adjusting for the confounder) while the
    // CPDAG completion without that adjustment has two: the mapping is refused with the typed
    // dimension reason rather than silently hydrating a subset or changing identification.
    let err = transferred
        .expect_err("a three-coefficient source cannot hydrate a two-coefficient completion");
    let msg = err.to_string();
    assert!(
        msg.contains("prior_dimension_mismatch")
            && msg.contains("posterior coefficient dimension 3 != expected n_coef 2"),
        "class-envelope transfer must refuse with the dimension mismatch: {msg}"
    );
}

#[test]
fn static_mediation_small_draw_requests_have_finite_uncertainty() {
    let (data, dag) = mediation_scm();
    let ctx = ExecutionContext::for_tests(180);
    let query = MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        Arc::from([VariableId::from_raw(1)]),
        MediationContrast::NaturalIndirect,
    );
    for n_draws in [0, 1] {
        let estimator = antecedent_estimate::BayesianGComputationAte::new().with_n_draws(n_draws);
        let err = antecedent_estimate::estimate_static_mediation_bayesian(
            &antecedent_estimate::StaticMediationProblem {
                data: &data,
                graph: &dag,
                query: &query,
                extra: &[],
            },
            antecedent_core::AssumptionSet::default(),
            &estimator,
            antecedent_core::IdentificationStatus::NonparametricallyIdentified,
            None,
            &ctx,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("n_draws >= 2"),
            "n_draws={n_draws} must refuse silent rewrite: {err}"
        );
    }
    let estimator = antecedent_estimate::BayesianGComputationAte::new().with_n_draws(2);
    let (_, posterior) = antecedent_estimate::estimate_static_mediation_bayesian(
        &antecedent_estimate::StaticMediationProblem {
            data: &data,
            graph: &dag,
            query: &query,
            extra: &[],
        },
        antecedent_core::AssumptionSet::default(),
        &estimator,
        antecedent_core::IdentificationStatus::NonparametricallyIdentified,
        None,
        &ctx,
    )
    .unwrap();
    assert_eq!(posterior.draws.n_draws, 2);
    assert!(posterior.summaries.sd.iter().all(|v| v.is_finite()));
}

#[test]
fn counterfactual_posterior_has_bayesian_bootstrap_variance() {
    // Saturated two-group Gaussian regression: each group mean has variance
    // empirical_var/(n_group+1) under Dirichlet row weights.
    let t: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect();
    let y: Vec<f64> = (0_usize..20).map(|i| (i % 10) as f64 + 5.0 * t[i]).collect();
    let data = TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())]).unwrap();
    let mut dag = Dag::with_variables(2);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = CounterfactualQuery::new(
        VariableId::from_raw(1),
        Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    )
    .with_control_level(0.0);
    let result = Study::tabular(data)
        .graph(dag)
        .query(CausalQuery::Counterfactual(query))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(3000)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(812))
        .unwrap();
    let posterior = result.posterior.as_ref().unwrap();
    let column = posterior.effect_column().unwrap();
    assert!((posterior.summaries.mean[column] - 5.0).abs() < 0.1);
    assert!(
        (posterior.summaries.sd[column].powi(2) - 1.5).abs() < 0.15,
        "variance {} should be 1.5, not inflated by a second resample",
        posterior.summaries.sd[column].powi(2)
    );
}

#[test]
fn inference_setter_can_return_to_frequentist_defaults() {
    let (data, dag) = mediation_scm();
    let study = Study::tabular(data)
        .graph(dag)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(2)))
        .inference(bayes())
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(1)).unwrap();
    assert!(result.posterior.is_none());
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("linear.adjustment.ate"));
}

#[test]
fn bayesian_derivative_refuses_unread_backend_and_likelihood() {
    let treatment: Vec<_> = (0..100).map(|i| f64::from(i) / 25.0).collect();
    let outcome: Vec<_> = treatment.iter().map(|x| 2.0 * x + 1.0).collect();
    let data = TabularData::from_f64_columns([
        ("treatment", treatment.as_slice()),
        ("outcome", outcome.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = || {
        CausalQuery::Response(ResponseQuery::new(ResponseFunctional::PointDerivative {
            outcome: VariableId::from_raw(1),
            treatment: VariableId::from_raw(0),
            at: 2.0,
            order: 1,
            scale: DerivativeScale::Identity,
        }))
    };
    let ctx = ExecutionContext::for_tests(18);
    let study = Study::tabular(data.clone())
        .graph(graph.clone())
        .query(query())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let error = study.prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap_err();
    assert_eq!(error.reason_code(), Some("option_not_applicable"));
    assert!(error.to_string().contains("Rubin Bayesian bootstrap"), "{error}");

    let error = Study::tabular(data)
        .graph(graph)
        .query(query())
        .inference(InferenceMode::Bayesian(
            BayesianConfig::laplace().likelihood(antecedent_prob::BayesLikelihood::BernoulliLogit),
        ))
        .refute(RefuteSuite::None)
        .build()
        .err()
        .expect("a derivative cannot use a Bernoulli likelihood");
    assert_eq!(error.reason_code(), Some("likelihood_not_supported"));
}

fn source_prior_bytes(data: &TabularData, dag: &Dag, treatment: u32, outcome: u32) -> Vec<u8> {
    let source = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(AverageEffectQuery::binary_ate(
            VariableId::from_raw(treatment),
            VariableId::from_raw(outcome),
        ))
        .inference(bayes())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(31))
        .unwrap();
    antecedent::io::encode_causal_posterior_bytes(source.posterior.as_ref().unwrap(), "source")
        .unwrap()
}

/// A transferred prior without a response-specific mapping is refused by the graph-posterior
/// response, not replaced by an isotropic prior on every atom.
#[test]
fn graph_posterior_response_refuses_a_transferred_prior_without_a_mapping() {
    let data = mixture_data(160);
    let mut dag = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let bytes = source_prior_bytes(&data, &dag, 0, 1);
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    });
    let ctx = ExecutionContext::for_tests(32);
    let config = BayesianConfig::conjugate().n_draws(32).prior_from_artifact(bytes, None);
    let built = Study::tabular(data)
        .graph_posterior(mixture_gp())
        .query(CausalQuery::Response(query))
        .inference(InferenceMode::Bayesian(config))
        .refute(RefuteSuite::None)
        .build();
    let error = match built {
        Err(error) => error,
        Ok(study) => study.run(&ctx).expect_err("an unmapped transfer must not run"),
    };
    assert!(error.to_string().contains("response-specific mapping"), "{error}");
}

/// The same refusal holds for the sealed static derivative response.
#[test]
fn derivative_response_refuses_a_transferred_prior_without_a_mapping() {
    let a: Vec<_> = (0..400)
        .map(|i| 2.0 + (f64::from(i) * 0.71).sin() + 0.2 * (f64::from(i) * 0.13).cos())
        .collect();
    let b: Vec<_> = (0..400).map(|i| (f64::from(i) * 1.13).cos()).collect();
    let y: Vec<_> = a.iter().zip(&b).map(|(a, b)| 5.0 + 2.0 * a - 0.5 * b).collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    for (from, to) in [(0, 2), (1, 2)] {
        dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let bytes = source_prior_bytes(&data, &dag, 0, 2);
    let functional = ResponseFunctional::PointDerivative {
        outcome: VariableId::from_raw(2),
        treatment: VariableId::from_raw(0),
        at: 2.0,
        order: 1,
        scale: DerivativeScale::Identity,
    };
    let ctx = ExecutionContext::for_tests(33);
    let config = BayesianConfig::conjugate().n_draws(32).prior_from_artifact(bytes, None);
    let built = Study::tabular(data)
        .graph(dag)
        .query(CausalQuery::Response(ResponseQuery::new(functional)))
        .inference(InferenceMode::Bayesian(config))
        .refute(RefuteSuite::None)
        .build();
    let error = match built {
        Err(error) => error,
        Ok(study) => study.run(&ctx).expect_err("an unmapped transfer must not run"),
    };
    assert!(error.to_string().contains("response-specific mapping"), "{error}");
}

/// Solve `a x = b` for a small dense system by Gaussian elimination with partial pivoting.
fn solve_small(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in (col + 1)..n {
            let factor = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let tail: f64 = ((row + 1)..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x
}

/// Posterior mean of the coefficients of a conjugate normal-inverse-gamma regression whose prior
/// is `beta | sigma^2 ~ N(m0, sigma^2 diag(v0))`: `(V0^-1 + X'X)^-1 (V0^-1 m0 + X'y)`. The
/// inverse-gamma hyperparameters do not enter the posterior mean of `beta`.
fn conjugate_posterior_mean(design: &[Vec<f64>], y: &[f64], m0: &[f64], v0: &[f64]) -> Vec<f64> {
    let p = m0.len();
    let mut normal = vec![vec![0.0; p]; p];
    let mut rhs = vec![0.0; p];
    for (row, outcome) in design.iter().zip(y) {
        for i in 0..p {
            rhs[i] += row[i] * outcome;
            for j in 0..p {
                normal[i][j] += row[i] * row[j];
            }
        }
    }
    for i in 0..p {
        normal[i][i] += 1.0 / v0[i];
        rhs[i] += m0[i] / v0[i];
    }
    solve_small(normal, rhs)
}

/// A transferred effect prior mapped onto the treatment slope reaches every atom of a graph
/// posterior, including atoms that adjust for different sets, and each atom's posterior mean of
/// the intervention level equals the independent conjugate calculation for its own design.
#[test]
fn graph_posterior_response_applies_a_mapped_transferred_prior_to_each_adjustment_set() {
    // Source: a small sample whose treatment effect is 0.5, so its posterior is informative but not
    // dominant over the 160 target rows, whose adjusted slope is 2 and unadjusted slope is 3.
    let source_columns = mixture_columns(32, 0.5);
    let mut dag = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let source = Study::tabular(mixture_table(&source_columns))
        .graph(dag)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(4_000)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(41))
        .unwrap();
    let posterior = source.posterior.as_ref().unwrap();
    let quantities = &posterior.draws.schema.quantities;
    let effect_column = posterior.effect_column().unwrap();
    let antecedent_prob::PosteriorQuantityKind::Effect { name: effect_name } =
        &quantities[effect_column]
    else {
        panic!("the source effect column is named");
    };
    let residual_column = quantities
        .iter()
        .position(|q| matches!(q, antecedent_prob::PosteriorQuantityKind::ResidualVariance))
        .unwrap();
    let (source_mean, source_sd, source_sigma2) = (
        posterior.summaries.mean[effect_column],
        posterior.summaries.sd[effect_column],
        posterior.summaries.mean[residual_column],
    );
    let bytes = antecedent::io::encode_causal_posterior_bytes(posterior, "source").unwrap();

    // Target: the binary-treatment contrast is one, so the slope prior is the effect prior. Other
    // coefficients keep the weakly informative baseline (mean 0, relative variance 100).
    let target_columns = mixture_columns(160, 2.0);
    let [t, y, z] = &target_columns;
    let slope_variance = (source_sd * source_sd) / source_sigma2;
    let level = |design: Vec<Vec<f64>>, m0: &[f64], v0: &[f64]| {
        let beta = conjugate_posterior_mean(&design, y, m0, v0);
        let z_mean = z.iter().sum::<f64>() / z.len() as f64;
        match beta.len() {
            2 => beta[0] + beta[1],
            _ => beta[0] + beta[1] + beta[2] * z_mean,
        }
    };
    let direct_design: Vec<Vec<f64>> = t.iter().map(|t| vec![1.0, *t]).collect();
    let adjusted_design: Vec<Vec<f64>> = t.iter().zip(z).map(|(t, z)| vec![1.0, *t, *z]).collect();
    let expected_direct =
        level(direct_design.clone(), &[0.0, source_mean], &[100.0, slope_variance]);
    let expected_adjusted =
        level(adjusted_design.clone(), &[0.0, source_mean, 0.0], &[100.0, slope_variance, 100.0]);
    let isotropic_direct = level(direct_design, &[0.0, 0.0], &[100.0, 100.0]);
    let isotropic_adjusted = level(adjusted_design, &[0.0, 0.0, 0.0], &[100.0, 100.0, 100.0]);
    // The transfer must be visible, or the comparison below would not tell it from no transfer.
    assert!(
        (expected_direct - isotropic_direct).abs() > 0.05,
        "{expected_direct} {isotropic_direct}"
    );
    assert!(
        (expected_adjusted - isotropic_adjusted).abs() > 0.05,
        "{expected_adjusted} {isotropic_adjusted}"
    );

    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    });
    let config = BayesianConfig::conjugate().n_draws(4_000).prior_from_artifact(
        bytes,
        Some(antecedent_io::PriorMapping::EffectFunctional {
            source_quantity: effect_name.to_string(),
        }),
    );
    let result = Study::tabular(mixture_table(&target_columns))
        .graph_posterior(mixture_gp())
        .query(CausalQuery::Response(query))
        .inference(InferenceMode::Bayesian(config))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(42))
        .unwrap();
    let mixture = result.structural_response.as_ref().unwrap();
    let level_of = |weight: f64| {
        let atom = mixture.atoms.iter().find(|atom| (atom.weight - weight).abs() < 1e-12).unwrap();
        let ResponseValue::Scalar(mean) = atom.value.as_ref().unwrap() else {
            panic!("an intervention level is a scalar");
        };
        *mean
    };
    // Atom weights 0.5 (no adjustment) and 0.3 (adjusting for the confounder) are the two
    // identified atoms; the 0.2 atom reverses the edge and is unidentified.
    // The posterior sd of a level here is about 0.02, so a mean of 4,000 draws is within a few
    // 1e-4 of the analytic posterior mean; the transfer shifts it by more than 0.05.
    for (weight, expected) in [(0.5, expected_direct), (0.3, expected_adjusted)] {
        let mean = level_of(weight);
        assert!(
            (mean - expected).abs() < 0.01,
            "atom {weight}: posterior mean {mean} vs independent conjugate calculation {expected}"
        );
    }
}

//! Crate-level tests: the day-1 flows through the root re-exports.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalSchemaBuilder, ExecutionContext, Intervention,
    InterventionalDistributionQuery, MeasurementSpec, PathSpecificEffectQuery, RoleHint,
    SmallRoleSet, StreamDomain, Value, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;

use super::*;
use crate::validate::PredictiveCheckKind;

fn scm() -> (TabularData, Dag, AverageEffectQuery) {
    let n = 200usize;
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "t",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let z: Vec<f64> = (0..n).map(|i| (i as f64) / n as f64).collect();
    let t: Vec<f64> = (0..n).map(|i| if z[i] > 0.5 { 1.0 } else { 0.0 }).collect();
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i] + 3.0 * z[i]).collect();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let mut dag = Dag::with_variables(3);
    // z -> t, z -> y, t -> y
    let z_id = DenseNodeId::from_raw(2);
    let t_id = DenseNodeId::from_raw(0);
    let y_id = DenseNodeId::from_raw(1);
    dag.insert_directed(z_id, t_id).unwrap();
    dag.insert_directed(z_id, y_id).unwrap();
    dag.insert_directed(t_id, y_id).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (TabularData::new(storage), dag, query)
}

#[test]
fn end_to_end_ate() {
    let (data, graph, query) = scm();
    let analysis = Study::tabular(data)
        .graph(graph)
        .query(query)
        .refute(RefuteSuite::PlaceboAndRcc)
        .bootstrap_replicates(30)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(3);
    let result = analysis.run(&ctx).unwrap();
    assert!((result.estimate.as_effect().unwrap().ate - 2.0).abs() < 1e-6);
    assert!(result.distribution().is_none());
    assert_eq!(result.refutations.len(), 2);
    assert!(!result.provenance.is_empty());
    assert!(!result.identification.derivation.steps.is_empty());

    let trace = result.analysis_trace_wire();
    assert_eq!(&*trace.method, "backdoor.adjustment");
    assert!(!trace.assumptions.is_empty());
    assert!(!trace.derivation.is_empty());
    let bytes = antecedent_io::to_cbor(&trace).unwrap();
    let round: antecedent_io::AnalysisTraceWire = antecedent_io::from_cbor(&bytes).unwrap();
    assert_eq!(round.method, trace.method);
}

#[test]
fn bayesian_ate_attaches_prior_and_posterior_predictive() {
    let (data, graph, query) = scm();
    let analysis = Study::tabular(data)
        .graph(graph)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .refute(RefuteSuite::PlaceboAndRcc)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let result = analysis.run(&ctx).unwrap();
    assert!(result.posterior.is_some());
    assert!(
        result.predictive_checks.iter().any(|c| c.kind == PredictiveCheckKind::Prior),
        "expected prior predictive check on Bayesian facade path"
    );
    assert!(
        result.predictive_checks.iter().any(|c| c.kind == PredictiveCheckKind::Posterior),
        "expected posterior predictive check on Bayesian facade path"
    );
    // Noise-free y = 1 + 2t + 3z adjusted for z: the posterior mean sits at the true ATE 2
    // (observed 2.0016), far from the unadjusted contrast 3.5.
    let fitted = result.posterior.as_ref().unwrap();
    let mean = fitted.summaries.mean[fitted.effect_column().unwrap()];
    assert!((mean - 2.0).abs() < 0.01, "posterior mean {mean}");
    let prior =
        result.predictive_checks.iter().find(|c| c.kind == PredictiveCheckKind::Prior).unwrap();
    assert!(prior.p_value.is_finite());
    assert!(prior.predictive_sd.is_finite());
    let post_ppc =
        result.predictive_checks.iter().find(|c| c.kind == PredictiveCheckKind::Posterior).unwrap();
    assert!(post_ppc.p_value.is_finite());
    assert!(post_ppc.predictive_sd.is_finite());
    assert!(result.refutations.iter().any(|r| r.refuter.as_ref() == "prior_predictive"));
    assert!(result.refutations.iter().any(|r| r.refuter.as_ref() == "posterior_predictive"));
}

// Bayesian graph-posterior discovery (exact / order / structure MCMC, CI-screened, DBN)
// is reachable from `Study` via `StudyBuilder::graph_posterior`, fed by a `GraphPosterior`
// the caller built directly from a free `discovery::discover_*` function (discovery is no
// longer run implicitly by the builder — see `StudyBuilder::graph_posterior`'s docs).

#[test]
fn bayesian_exact_dag_posterior_effect_envelope() {
    let (data, _graph, query) = scm();
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let ctx = ExecutionContext::for_tests(1);
    let gp = crate::discovery::discover_exact_dag_posterior(
        &data,
        &vars,
        &crate::discovery::BayesianDiscoverParams::default(),
        &ctx,
    )
    .unwrap();
    let analysis = Study::tabular(data)
        .graph_posterior(gp)
        .query(query)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(80).prior_scale(100.0),
        ))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let result = analysis.run(&ctx).unwrap();
    let post = result.posterior.expect("mixture posterior");
    assert!((0.0..=1.0).contains(&post.unidentified_mass));
    let eq = post.effect_column().unwrap();
    // Atoms either adjust for z (effect 2) or not (the unadjusted contrast 3 * (0.75 - 0.25) + 2
    // = 3.5), so the mixture mean lies between the two closed forms; both are exact on this data.
    let mean = post.summaries.mean[eq];
    assert!((1.99..=3.51).contains(&mean), "mixture mean {mean} outside [2, 3.5]");
    assert!(post.summaries.sd[eq] > 0.0 && post.summaries.sd[eq].is_finite());
    assert!(post.draws.n_draws > 0);
}

/// Every posterior producer must tag itself, and that tag must survive into the
/// plan record — which is surfaced to Python and serialized into CBOR artifacts.
///
/// This is a regression gate. A refactor once made every posterior algorithm record
/// the generic label `"graph_posterior"`, and the whole suite stayed green because
/// nothing anywhere asserted `discovery_algorithm`. An analysis that cannot say
/// which structure-learning algorithm produced its posterior is not reproducible.
#[test]
fn graph_posterior_records_its_real_algorithm() {
    let (data, _graph, query) = scm();
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let ctx = ExecutionContext::for_tests(1);
    let params = crate::discovery::BayesianDiscoverParams::default();

    let exact =
        crate::discovery::discover_exact_dag_posterior(&data, &vars, &params, &ctx).unwrap();
    assert_eq!(exact.algorithm().map(|name| name.as_ref()), Some("exact_dag_posterior"));

    let schedule = crate::discovery::GraphMcmcSchedule {
        n_chains: 2,
        n_warmup: 20,
        n_draws: 40,
        ..Default::default()
    };
    let order =
        crate::discovery::discover_order_mcmc(&data, &vars, &params, &schedule, false, &ctx)
            .unwrap();
    assert_eq!(
        order.algorithm().map(|name| name.as_ref()),
        Some("order_mcmc"),
        "order MCMC must not report a generic posterior label"
    );

    // End-to-end: the tag reaches the plan record, not just the posterior struct.
    let plan = Study::tabular(data)
        .graph_posterior(order)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(40)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .plan(&ctx)
        .unwrap();
    assert_eq!(plan.logical.record.discovery_algorithm.as_deref(), Some("order_mcmc"));
}

#[test]
fn graph_posterior_average_effect_accepts_frequentist() {
    let (data, _graph, query) = scm();
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let ctx = ExecutionContext::for_tests(1);
    let gp = crate::discovery::discover_exact_dag_posterior(
        &data,
        &vars,
        &crate::discovery::BayesianDiscoverParams::default(),
        &ctx,
    )
    .unwrap();
    let result = Study::tabular(data)
        .graph_posterior(gp)
        .query(query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .build()
        .expect("AverageEffect × graph_posterior × Frequentist is licensed")
        .run(&ctx)
        .unwrap();
    // Atoms disagree on the estimand (adjust for z: 2.0; do not: 3 * (0.75 - 0.25) + 2 = 3.5), so
    // the pooled scalar is withheld rather than averaged; each retained atom value is one of the
    // two exact closed forms on this noise-free data.
    assert!(result.estimate.as_effect().unwrap().ate.is_nan());
    assert!(result.diagnostics.iter().any(|d| {
        d.code.as_ref() == "estimate.graph_posterior.envelope"
            && d.message.contains("scalar withheld: contributing atoms disagree")
    }));
    let mixture = result.structural_response.as_ref().expect("per-atom values are retained");
    let values: Vec<f64> = mixture
        .atoms
        .iter()
        .filter_map(|atom| match atom.value {
            Some(antecedent_core::ResponseValue::Scalar(v)) => Some(v),
            _ => None,
        })
        .collect();
    assert!(!values.is_empty());
    for v in &values {
        assert!((v - 3.5).abs() < 1e-9 || (v - 2.0).abs() < 1e-9, "atom value {v}");
    }
    assert!(values.iter().any(|v| (v - 3.5).abs() < 1e-9));
    assert!(values.iter().any(|v| (v - 2.0).abs() < 1e-9));
}

#[test]
fn graph_and_graph_posterior_together_conflicts() {
    let (data, graph, query) = scm();
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let ctx = ExecutionContext::for_tests(1);
    let gp = crate::discovery::discover_exact_dag_posterior(
        &data,
        &vars,
        &crate::discovery::BayesianDiscoverParams::default(),
        &ctx,
    )
    .unwrap();
    let err = Study::tabular(data).graph(graph).graph_posterior(gp).query(query).build();
    match err {
        Err(CausalError::Conflict { what, .. }) => assert_eq!(what, "graph"),
        other => panic!("expected CausalError::Conflict, got {other:?}"),
    }
}

/// `(z, t, y, count)` rows over columns `t, y, z` with `z -> t`, `z -> y`,
/// `t -> y`; returns the Frequentist `P(y | do(t = 1))` study result.
fn run_binary_distribution(combos: &[(f64, f64, f64, usize)]) -> StudyResult {
    let mut b = CausalSchemaBuilder::new();
    for name in ["t", "y", "z"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let mut t_vals = Vec::new();
    let mut y_vals = Vec::new();
    let mut z_vals = Vec::new();
    for &(z, t, y, count) in combos {
        for _ in 0..count {
            z_vals.push(z);
            t_vals.push(t);
            y_vals.push(y);
        }
    }
    let n = t_vals.len();
    let column = |raw: u32, values: Vec<f64>| {
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(raw),
                Arc::from(values),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        )
    };
    let cols = vec![column(0, t_vals), column(1, y_vals), column(2, z_vals)];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = InterventionalDistributionQuery::new(
        VariableId::from_raw(1),
        [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
    );
    let analysis = Study::tabular(data)
        .graph(dag)
        .query(CausalQuery::Distribution(query))
        .identifier(IdentifierId::GeneralId)
        .estimator(EstimatorId::FunctionalDistribution)
        .build()
        .unwrap();
    analysis.run(&ExecutionContext::for_tests(0)).unwrap()
}

#[test]
fn end_to_end_interventional_distribution() {
    // Discrete confounding table matching functional_distribution unit test.
    let result = run_binary_distribution(&[
        (0.0, 0.0, 0.0, 21),
        (0.0, 0.0, 1.0, 9),
        (0.0, 1.0, 0.0, 4),
        (0.0, 1.0, 1.0, 16),
        (1.0, 0.0, 0.0, 12),
        (1.0, 0.0, 1.0, 3),
        (1.0, 1.0, 0.0, 14),
        (1.0, 1.0, 1.0, 21),
    ]);
    // Frozen expectation: read from the conformance fixture rather than
    // duplicating its numbers here. The fixture was previously loaded by
    // nothing — this test hardcoded the same values, so the two could
    // drift apart with neither noticing.
    let fx: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/interventional_distribution/expected.json"
    ))
    .unwrap();
    assert_eq!(fx["method"], "general.id");
    assert_eq!(fx["estimator"], "functional.distribution");
    let (mean, tol) = (fx["mean"].as_f64().unwrap(), fx["tolerance"].as_f64().unwrap());
    let dist = result.distribution().expect("distribution payload");
    assert!((dist.mean - mean).abs() < tol, "mean={}", dist.mean);
    assert!(result.estimate.as_effect().unwrap().ate.is_finite());
    // Every atom probability carries a bounded interval inside [0, 1], and
    // the binary mean's interval is the `Y = 1` atom's.
    assert_eq!(dist.atom_uncertainty.len(), dist.atoms.len());
    for (atom, u) in dist.atoms.iter().zip(dist.atom_uncertainty.iter()) {
        let (lo, hi) = u.interval.bounds().expect("interior atoms are bounded");
        assert!(0.0 < lo && lo < atom.probability && atom.probability < hi && hi < 1.0);
    }
    let (lo, hi) = dist.mean_interval.and_then(|i| i.bounds()).expect("mean interval");
    assert!(lo < dist.mean && dist.mean < hi);
    assert!(
        !result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.distribution.interval_unavailable")
    );
}

#[test]
fn boundary_interventional_probability_reports_unavailable_interval() {
    // `y` never occurs under `t = 1`: `P(y = 1 | do(t = 1)) = 0` and
    // `P(y = 0 | do(t = 1)) = 1`. Neither can carry an interval.
    let result = run_binary_distribution(&[
        (0.0, 0.0, 0.0, 20),
        (0.0, 0.0, 1.0, 10),
        (0.0, 1.0, 0.0, 20),
        (1.0, 0.0, 0.0, 12),
        (1.0, 0.0, 1.0, 8),
        (1.0, 1.0, 0.0, 30),
    ]);
    let dist = result.distribution().expect("distribution payload");
    assert_eq!(dist.atom_uncertainty.len(), dist.atoms.len());
    for u in dist.atom_uncertainty.iter() {
        assert_eq!(
            u.interval,
            antecedent_estimate::ProbabilityInterval::Unavailable(
                antecedent_estimate::ProbabilityIntervalUnavailable::BoundaryEstimate
            )
        );
    }
    assert_eq!(
        dist.mean_interval,
        Some(antecedent_estimate::ProbabilityInterval::Unavailable(
            antecedent_estimate::ProbabilityIntervalUnavailable::BoundaryEstimate
        ))
    );
    let diagnostic = result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "estimate.distribution.interval_unavailable")
        .expect("unavailable intervals are disclosed");
    assert!(diagnostic.message.contains("boundary_estimate"), "{}", diagnostic.message);
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "builds the discrete chain, runs the natural path-specific effect and asserts it against the exact path decomposition"
)]
fn end_to_end_path_specific_natural_effect() {
    // Discrete chain T → M → Y (no direct edge): path via M is the total effect.
    let mut b = CausalSchemaBuilder::new();
    for name in ["t", "m", "y"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    // Complete binary chain: every (t,m,y) cell is observed so the ID
    // plug-in can evaluate required CPT entries. E[Y|T=1]−E[Y|T=0] = 0.3.
    let mut t_vals = Vec::new();
    let mut m_vals = Vec::new();
    let mut y_vals = Vec::new();
    for (t, m, y, count) in [
        (0.0, 0.0, 0.0, 40),
        (0.0, 0.0, 1.0, 10),
        (0.0, 1.0, 0.0, 10),
        (0.0, 1.0, 1.0, 40),
        (1.0, 0.0, 0.0, 10),
        (1.0, 0.0, 1.0, 10),
        (1.0, 1.0, 0.0, 10),
        (1.0, 1.0, 1.0, 70),
    ] {
        for _ in 0..count {
            t_vals.push(t);
            m_vals.push(m);
            y_vals.push(y);
        }
    }
    let n = t_vals.len();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(t_vals),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(1),
                Arc::from(m_vals),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(2),
                Arc::from(y_vals),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let query = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
        .with_path_nodes([VariableId::from_raw(1)]);
    let analysis = Study::tabular(data)
        .graph(dag)
        .query(CausalQuery::PathSpecific(query))
        .identifier(IdentifierId::PathSpecificNatural)
        .estimator(EstimatorId::FunctionalEffect)
        .build()
        .unwrap();
    let result = analysis.run(&ExecutionContext::for_tests(0)).unwrap();
    // Frozen expectations: both path_specific_natural fixtures were
    // previously loaded by nothing while this test hardcoded the same
    // values (found by the evidence-reachability gate's word-boundary
    // scan — the fixture basename matched this function's *name*).
    let ctxfx: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/context/path_specific_natural/expected.json"
    ))
    .unwrap();
    let (ate, tol) = (ctxfx["ate"].as_f64().unwrap(), ctxfx["tolerance"].as_f64().unwrap());
    assert!(
        (result.estimate.as_effect().unwrap().ate - ate).abs() < tol,
        "ate={}",
        result.estimate.as_effect().unwrap().ate
    );
    assert_eq!(result.estimand.method.as_ref(), ctxfx["method"].as_str().unwrap());
    let idfx: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/identify/path_specific_natural/expected.json"
    ))
    .unwrap();
    assert_eq!(idfx["method"], "path_specific.natural");
    let want_status: String = idfx["status"]
        .as_str()
        .unwrap()
        .split('_')
        .map(|w| {
            let mut c = w.chars();
            c.next().map(|f| f.to_ascii_uppercase().to_string() + c.as_str()).unwrap_or_default()
        })
        .collect();
    assert_eq!(format!("{:?}", result.identification.status), want_status);
}

/// Confounded SCM: `Z ~ N(0,1)`, `T ~ Bernoulli(logit(-0.5 + Z))`, `Y = 2T + Z + noise`.
/// True ATE = 2; OLS-on-observables is biased here unless `Z` is adjusted for, so this
/// exercises the propensity path independent of the linear-adjustment default.
fn confounded_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng =
        ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Execute, 0x1234_u64);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let logit = -0.5 + zi;
        let p = 1.0 / (1.0 + (-logit).exp());
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        let noise = standard_normal(&mut rng) * 0.5;
        z[i] = zi;
        t[i] = ti;
        y[i] = 2.0 * ti + zi + noise;
    }
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "t",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let mut dag = Dag::with_variables(3);
    let z_id = DenseNodeId::from_raw(2);
    let t_id = DenseNodeId::from_raw(0);
    let y_id = DenseNodeId::from_raw(1);
    dag.insert_directed(z_id, t_id).unwrap();
    dag.insert_directed(z_id, y_id).unwrap();
    dag.insert_directed(t_id, y_id).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (TabularData::new(storage), dag, query)
}

#[test]
fn end_to_end_propensity_weighting_recovers_confounded_effect() {
    // Dual of python/tests/test_analyze_ate_estimate.py
    // (`test_analyze_propensity_weighting_recovers_ate_and_overlap`): true ATE=2;
    // Rust band |ate−2|<0.06 (observed 0.015); the unadjusted contrast is ~2.8, so 0.06 still
    // separates adjusted from unadjusted by a wide margin.
    let (data, graph, query) = confounded_scm(800, 1);
    let analysis = Study::tabular(data)
        .graph(graph)
        .query(query)
        .identifier(IdentifierId::BackdoorAdjustment)
        .estimator(EstimatorId::PropensityWeighting)
        .bootstrap_replicates(30)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(11);
    let result = analysis.run(&ctx).unwrap();
    assert!(
        (result.estimate.as_effect().unwrap().ate - 2.0).abs() < 0.06,
        "ate={}",
        result.estimate.as_effect().unwrap().ate
    );
    // Placebo/RCC refuters are hardwired to LinearAdjustmentAte and are skipped for
    // every non-default estimator (see `Study::execute_static`).
    assert!(result.refutations.is_empty());
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("propensity.weighting"));
}

/// `Z ∈ {0,1} → T → Y` with `U` confounding `T`–`Y` (unobserved, not in the graph).
/// True structural effect = 2.0.
fn iv_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng =
        ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Execute, 0x1E71_u64);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = (i % 2) as f64;
        let u = standard_normal(&mut rng);
        let ti = 0.5 * zi + u + 0.1 * standard_normal(&mut rng);
        let yi = 2.0 * ti + u + 0.1 * standard_normal(&mut rng);
        z[i] = zi;
        t[i] = ti;
        y[i] = yi;
    }
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "t",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let mut dag = Dag::with_variables(3);
    let z_id = DenseNodeId::from_raw(2);
    let t_id = DenseNodeId::from_raw(0);
    let y_id = DenseNodeId::from_raw(1);
    dag.insert_directed(z_id, t_id).unwrap();
    dag.insert_directed(t_id, y_id).unwrap();
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    (TabularData::new(storage), dag, query)
}

#[test]
fn end_to_end_iv_two_stage_least_squares() {
    let (data, graph, query) = iv_scm(4000, 5);
    let analysis = Study::tabular(data)
        .graph(graph)
        .query(query)
        .identifier(IdentifierId::Iv)
        .estimator(EstimatorId::Iv2Sls)
        .bootstrap_replicates(30)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(21);
    let result = analysis.run(&ctx).unwrap();
    assert!(
        (result.estimate.as_effect().unwrap().ate - 2.0).abs() < 0.1,
        "ate={}",
        result.estimate.as_effect().unwrap().ate
    );
    assert!(result.refutations.is_empty());
}

#[test]
fn end_to_end_temporal_effect() {
    use antecedent_core::{Lag, TemporalEffectQuery, TemporalPolicy};
    use antecedent_data::{SamplingRegularity, TimeIndex, TimeSeriesData};
    use antecedent_graph::{TemporalDag, ensure_lagged};

    let n = 250usize;
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "pressure",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "defect",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = ((t as f64) * 0.05).sin();
        y[t] = 0.75 * x[t - 1];
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(x), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let series = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let mut g = TemporalDag::empty();
    let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(x1, y0).unwrap();
    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);
    let analysis =
        Study::series(series).graph(g).temporal_query(q).bootstrap_replicates(0).build().unwrap();
    let ctx = ExecutionContext::for_tests(7);
    let result = analysis.run(&ctx).unwrap();
    assert!(
        (result.estimate.as_effect().unwrap().ate - 0.75).abs() < 1e-9,
        "ate={}",
        result.estimate.as_effect().unwrap().ate
    );
    assert_eq!(&*result.logical_plan.plan_id, "temporal_effect");
    assert!(result.physical_plan.estimated_peak_memory_bytes.is_some());
    assert!(result.physical_plan.estimated_copy_bytes.is_some());
    assert!(!result.physical_plan.task_schedule.is_empty());
    assert!(!result.physical_plan.materializations.is_empty());

    let plan = analysis.compile(&ctx).unwrap();
    assert!(plan.temporal_graph().is_some());
    assert_eq!(plan.record.batch_size, Some(250));
}

#[test]
fn every_interval_method_has_a_record_or_code() {
    for method in antecedent_core::IntervalMethod::ALL {
        let has_record = crate::coverage_records_data::RECORDS
            .iter()
            .any(|row| row.interval_method == method.as_str());
        let listed = crate::coverage_records_data::INTERVAL_METHOD_REASONS
            .iter()
            .any(|(item, reason)| *item == method && (has_record || reason.is_some()));
        assert!(
            has_record || listed,
            "IntervalMethod::{method:?} needs a coverage record or INTERVAL_METHOD_REASONS code"
        );
    }
}

//! Unified static/temporal `CausalAnalysis` facade (identify → estimate → refute).
//!
//! # Quick start
//!
//! ```
//! use causal::prelude::*;
//! use causal::RefuteSuite;
//!
//! let schema = CausalSchemaBuilder::new()
//!     .continuous("t")
//!     .treatment()
//!     .continuous("y")
//!     .outcome()
//!     .continuous("z")
//!     .context()
//!     .build()
//!     .unwrap();
//! let data = TabularData::from_f64_columns([
//!     ("t", &[0.0_f64, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0][..]),
//!     ("y", &[1.0_f64, 3.1, 1.2, 2.9, 0.9, 3.0, 1.1, 3.2][..]),
//!     ("z", &[0.0_f64, 1.0, 0.2, 0.8, 0.1, 0.9, 0.3, 0.7][..]),
//! ])
//! .unwrap();
//! let dag = Dag::from_named_edges(&schema, &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
//! let q = AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
//! let result = CausalAnalysis::builder()
//!     .data(data)
//!     .graph(dag)
//!     .query(q)
//!     .refute(RefuteSuite::None)
//!     .bootstrap_replicates(0)
//!     .build()
//!     .unwrap()
//!     .run(&ExecutionContext::for_tests(1))
//!     .unwrap();
//! assert!(result.effect().is_finite());
//! ```
//!
//! Prefer [`prelude`] for day-1 imports. Component crates remain available for stage-specific work.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::missing_errors_doc, clippy::missing_panics_doc)]

pub mod analysis;
pub mod callback_plan;
pub mod design;
pub mod discovery;
pub mod discovery_defaults;
pub mod error;
pub mod gcm;
pub mod inference;
pub mod options;
pub mod planner;
pub mod prelude;
pub mod result;
pub mod review;
pub mod state;
pub mod strategy_table;

pub mod estimate;
pub mod graph;
pub mod identify;
pub mod io;
pub mod query;
pub mod validate;

// --- Day-1 crate-root surface (stage depth lives under modules) ---
pub use analysis::{
    AnalysisStageEvent, BatchAnalysis, CausalAnalysis, CausalAnalysisBuilder, ComputeBudget,
    LatencyMode, PreparedAnalysis, RdConfig, RefuteSuite, StageResultSink,
};
#[allow(deprecated)]
pub use error::AnalysisError;
pub use error::CausalError;
pub use estimate::{CausalPosterior, EffectEstimate, EstimatorId, IdentifierId};
pub use graph::{Dag, DenseNodeId, TemporalDag};
pub use inference::{BayesianConfig, InferenceMode};
pub use options::{DiscoveryAccept, FdrControl};
pub use planner::{CompiledAnalysis, GraphInput};
pub use query::*;
pub use result::CausalAnalysisResult;

// Strategy helpers and peer APIs: use `causal::estimate`, `causal::discovery`,
// `causal::gcm`, `causal::io`, `causal::design`, `causal::state`, `causal::graph`,
// `causal::identify`, `causal::review`, `causal::validate`, `causal::planner`.

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        AverageEffectQuery, CausalQuery, CausalSchemaBuilder, ExecutionContext, Intervention,
        InterventionalDistributionQuery, MeasurementSpec, PathSpecificEffectQuery, RoleHint,
        SmallRoleSet, Value, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use causal_graph::{Dag, DenseNodeId};
    use causal_kernels::standard_normal;

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
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
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
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        (TabularData::new(storage), dag, query)
    }

    #[test]
    fn end_to_end_ate() {
        let (data, graph, query) = scm();
        let analysis = CausalAnalysis::builder()
            .data(data)
            .graph(graph)
            .query(query)
            .refute(RefuteSuite::PlaceboAndRcc)
            .bootstrap_replicates(30)
            .build()
            .unwrap();
        let ctx = ExecutionContext::for_tests(3);
        let result = analysis.run(&ctx).unwrap();
        assert!((result.estimate.ate - 2.0).abs() < 1e-6);
        assert!(result.distribution.is_none());
        assert_eq!(result.refutations.len(), 2);
        assert!(!result.provenance.is_empty());
        assert!(!result.identification.derivation.steps.is_empty());

        let trace = result.analysis_trace_wire();
        assert_eq!(&*trace.method, "backdoor.adjustment");
        assert!(!trace.assumptions.is_empty());
        assert!(!trace.derivation.is_empty());
        let bytes = causal_io::to_cbor(&trace).unwrap();
        let round: causal_io::AnalysisTraceWire = causal_io::from_cbor(&bytes).unwrap();
        assert_eq!(round.method, trace.method);
    }

    #[test]
    fn bayesian_ate_attaches_prior_and_posterior_predictive() {
        let (data, graph, query) = scm();
        let analysis = CausalAnalysis::builder()
            .data(data)
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
        let prior =
            result.predictive_checks.iter().find(|c| c.kind == PredictiveCheckKind::Prior).unwrap();
        assert!(prior.p_value.is_finite());
        assert!(prior.predictive_sd.is_finite());
        let post_ppc = result
            .predictive_checks
            .iter()
            .find(|c| c.kind == PredictiveCheckKind::Posterior)
            .unwrap();
        assert!(post_ppc.p_value.is_finite());
        assert!(post_ppc.predictive_sd.is_finite());
        assert!(result.refutations.iter().any(|r| r.refuter.as_ref() == "prior_predictive"));
        assert!(result.refutations.iter().any(|r| r.refuter.as_ref() == "posterior_predictive"));
    }

    #[test]
    fn bayesian_exact_dag_posterior_effect_envelope() {
        let (data, _graph, query) = scm();
        let analysis = CausalAnalysis::builder()
            .data(data)
            .discover_exact_dag_posterior()
            .query(query)
            .inference(InferenceMode::Bayesian(
                BayesianConfig::conjugate().n_draws(80).prior_scale(100.0),
            ))
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let result = analysis.run(&ctx).unwrap();
        let post = result.posterior.expect("mixture posterior");
        assert!((0.0..=1.0).contains(&post.unidentified_mass));
        let eq = post.effect_column().unwrap();
        assert!(post.summaries.mean[eq].is_finite());
        assert!(post.summaries.sd[eq].is_finite());
        assert!(post.draws.n_draws > 0);
    }

    #[test]
    fn graph_posterior_discovery_rejects_frequentist() {
        let (data, _graph, query) = scm();
        let err = CausalAnalysis::builder()
            .data(data)
            .discover_exact_dag_posterior()
            .query(query)
            .inference(InferenceMode::Frequentist)
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .compile(&ExecutionContext::for_tests(1))
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Bayesian") || msg.contains("graph-posterior"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn end_to_end_interventional_distribution() {
        // Discrete confounding table matching functional_distribution unit test.
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
        let combos = [
            (0.0, 0.0, 0.0, 21),
            (0.0, 0.0, 1.0, 9),
            (0.0, 1.0, 0.0, 4),
            (0.0, 1.0, 1.0, 16),
            (1.0, 0.0, 0.0, 12),
            (1.0, 0.0, 1.0, 3),
            (1.0, 1.0, 0.0, 14),
            (1.0, 1.0, 1.0, 21),
        ];
        let mut t_vals = Vec::new();
        let mut y_vals = Vec::new();
        let mut z_vals = Vec::new();
        for (z, t, y, count) in combos {
            for _ in 0..count {
                z_vals.push(z);
                t_vals.push(t);
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
                    Arc::from(y_vals),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z_vals),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut dag = Dag::with_variables(3);
        dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let query = InterventionalDistributionQuery::new(
            VariableId::from_raw(1),
            [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
        );
        let analysis = CausalAnalysis::builder()
            .data(data)
            .graph(dag)
            .query(CausalQuery::Distribution(query))
            .identifier(IdentifierId::GeneralId)
            .estimator(EstimatorId::FunctionalDistribution)
            .build()
            .unwrap();
        let result = analysis.run(&ExecutionContext::for_tests(0)).unwrap();
        let dist = result.distribution.expect("distribution payload");
        assert!((dist.mean - 0.7).abs() < 0.05, "mean={}", dist.mean);
        assert!(result.estimate.ate.is_finite());
    }

    #[test]
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
        // Deterministic: T~Bernoulli(0.5), M=T, Y=M  → ATE = 1
        let mut t_vals = Vec::new();
        let mut m_vals = Vec::new();
        let mut y_vals = Vec::new();
        for t in [0.0, 1.0] {
            for _ in 0..50 {
                t_vals.push(t);
                m_vals.push(t);
                y_vals.push(t);
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
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut dag = Dag::with_variables(3);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let query =
            PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
                .with_path_nodes([VariableId::from_raw(1)]);
        let analysis = CausalAnalysis::builder()
            .data(data)
            .graph(dag)
            .query(CausalQuery::PathSpecific(query))
            .identifier(IdentifierId::PathSpecificNatural)
            .estimator(EstimatorId::FunctionalEffect)
            .build()
            .unwrap();
        let result = analysis.run(&ExecutionContext::for_tests(0)).unwrap();
        assert!((result.estimate.ate - 1.0).abs() < 0.05, "ate={}", result.estimate.ate);
        assert_eq!(result.estimand.method.as_ref(), "path_specific.natural");
    }

    /// Confounded SCM: `Z ~ N(0,1)`, `T ~ Bernoulli(logit(-0.5 + Z))`, `Y = 2T + Z + noise`.
    /// True ATE = 2; OLS-on-observables is biased here unless `Z` is adjusted for, so this
    /// exercises the propensity path independent of the linear-adjustment default.
    fn confounded_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x1234_u64);
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
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
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
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        (TabularData::new(storage), dag, query)
    }

    #[test]
    fn end_to_end_propensity_weighting_recovers_confounded_effect() {
        // Dual of python/tests/test_analyze_ate_estimate.py
        // (`test_analyze_propensity_weighting_recovers_ate_and_overlap`): true ATE=2;
        // Rust band |ate−2|<0.3; shared cross-language floor is 0.4.
        let (data, graph, query) = confounded_scm(800, 1);
        let analysis = CausalAnalysis::builder()
            .data(data)
            .graph(graph)
            .query(query)
            .identifier("backdoor.adjustment")
            .estimator("propensity.weighting")
            .bootstrap_replicates(30)
            .build()
            .unwrap();
        let ctx = ExecutionContext::for_tests(11);
        let result = analysis.run(&ctx).unwrap();
        assert!((result.estimate.ate - 2.0).abs() < 0.3, "ate={}", result.estimate.ate);
        // Placebo/RCC refuters are hardwired to LinearAdjustmentAte and are skipped for
        // every non-default estimator (see `CausalAnalysis::execute_static`).
        assert!(result.refutations.is_empty());
        assert_eq!(result.logical_plan.estimator.as_deref(), Some("propensity.weighting"));
    }

    /// `Z ∈ {0,1} → T → Y` with `U` confounding `T`–`Y` (unobserved, not in the graph).
    /// True structural effect = 2.0.
    fn iv_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x1E71_u64);
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
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
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
        let query = AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            0.0,
            1.0,
        );
        (TabularData::new(storage), dag, query)
    }

    #[test]
    fn end_to_end_iv_two_stage_least_squares() {
        let (data, graph, query) = iv_scm(4000, 5);
        let analysis = CausalAnalysis::builder()
            .data(data)
            .graph(graph)
            .query(query)
            .identifier("iv")
            .estimator("iv.2sls")
            .bootstrap_replicates(30)
            .build()
            .unwrap();
        let ctx = ExecutionContext::for_tests(21);
        let result = analysis.run(&ctx).unwrap();
        assert!((result.estimate.ate - 2.0).abs() < 0.6, "ate={}", result.estimate.ate);
        assert!(result.refutations.is_empty());
    }

    #[test]
    fn end_to_end_temporal_effect() {
        use causal_core::{Lag, TemporalEffectQuery, TemporalPolicy};
        use causal_data::{SamplingRegularity, TimeIndex, TimeSeriesData};
        use causal_graph::{TemporalDag, ensure_lagged};

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
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
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
        let analysis = CausalAnalysis::builder()
            .series(series)
            .temporal_graph(g)
            .temporal_query(q)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let ctx = ExecutionContext::for_tests(7);
        let result = analysis.run(&ctx).unwrap();
        assert!((result.estimate.ate - 0.75).abs() < 0.08, "ate={}", result.estimate.ate);
        assert_eq!(&*result.logical_plan.plan_id, "temporal_effect");
        assert!(result.physical_plan.estimated_peak_memory_bytes.is_some());
        assert!(result.physical_plan.estimated_copy_bytes.is_some());
        assert!(!result.physical_plan.task_schedule.is_empty());
        assert!(!result.physical_plan.materializations.is_empty());

        let compiled = analysis.compile(&ctx).unwrap();
        match compiled {
            CompiledAnalysis::Ready(plan) => {
                assert!(plan.temporal_graph().is_some());
                assert_eq!(plan.record.batch_size, Some(250));
            }
            CompiledAnalysis::ReviewRequired(_)
            | CompiledAnalysis::ReviewRequiredCpdag(_)
            | CompiledAnalysis::ReviewRequiredStaticCpdag(_)
            | CompiledAnalysis::ReviewRequiredStaticDag(_)
            | CompiledAnalysis::ReviewRequiredPag(_)
            | CompiledAnalysis::ReviewRequiredStaticPag(_) => {
                panic!("expected Ready")
            }
        }
    }
}

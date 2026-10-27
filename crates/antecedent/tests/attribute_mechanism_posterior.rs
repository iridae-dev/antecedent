//! The posterior over the mechanisms of a fitted-model attribution, beside its frequentist
//! counterpart.
//!
//! * A distribution-change or root-cause attribution under Bayesian inference refits every
//!   mechanism per population under Dirichlet row weights and publishes the total and its
//!   components as a posterior; under Frequentist inference a replicate count publishes the
//!   mechanism-refit bootstrap standard error of the total. Both centre on the fitted
//!   model's answer.
//! * An intrinsic-influence attribution under Bayesian inference publishes the posterior of
//!   every source's variance share from Dirichlet-weighted refits, the families held at the
//!   ones selected on the data.
//! * The other fitted-model targets keep reporting the fitted model's answer and say so.
//! * The posterior route is explicitly unlicensed: it carries no coverage record.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{BayesianConfig, CompiledExecution, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AnomalyAttributionQuery, AttributeQuery, AttributeTarget, AttributionComponents, CausalQuery,
    ChangeAttributionQuery, ChangeEstimator, ChangeMeasure, ExecutionContext, InfluenceAllocation,
    IntrinsicInfluenceTarget, Lag, MechanismChangeQuery, MechanismChangeTest, NoiseDeclaration,
    PopulationSelector, UnitChangeQuery, VariableId,
};
use antecedent_core::{
    Intervention, InterventionSequence, SequencedIntervention, SimulateQuery, SimulationTarget,
    TemporalPolicy, Value,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_graph::{Dag, DenseNodeId, TemporalDag, ensure_lagged};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0xA77_B057)
}

fn on_large_stack(name: &str, body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.into())
        .stack_size(32 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

/// Deterministic standard-normal stream (xorshift64* with Box-Muller).
struct Stream(u64);

impl Stream {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// Two periods of `x -> y` with noise; the intercept of `y` shifts by 5 in the second.
fn two_period_table() -> TabularData {
    let mut stream = Stream(0x2F1E_77A1);
    let x: Vec<f64> = (0..200).map(|_| stream.normal()).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let shift = if i < 100 { 1.0 } else { 6.0 };
            shift + 2.0 * x + 0.5 * stream.normal()
        })
        .collect();
    TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap()
}

fn two_node_dag() -> Dag {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph
}

/// `z -> t -> y`, `z -> y` with Gaussian noise.
fn chain_table() -> TabularData {
    let mut stream = Stream(0x91C3_0B5D);
    let z: Vec<f64> = (0..300).map(|_| stream.normal()).collect();
    let t: Vec<f64> = z.iter().map(|z| 0.8 * z + stream.normal()).collect();
    let y: Vec<f64> = t.iter().zip(&z).map(|(t, z)| 1.5 * t + 0.6 * z + stream.normal()).collect();
    TabularData::from_f64_columns([("z", z.as_slice()), ("t", t.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

fn chain_dag() -> Dag {
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(0, 1), (1, 2), (0, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn change_target(rooted: bool) -> CausalQuery {
    let change = ChangeAttributionQuery::new(
        VariableId::from_raw(1),
        PopulationSelector::TimeRange { start: 0, end: 100 },
        PopulationSelector::TimeRange { start: 100, end: 200 },
    );
    let estimator = ChangeEstimator::Sampled {
        measure: ChangeMeasure::MeanDifference,
        n_samples: 400,
        seed: 3,
    };
    CausalQuery::Attribute(AttributeQuery::new(if rooted {
        AttributeTarget::RootCauses { change, estimator }
    } else {
        AttributeTarget::DistributionChange { change, estimator }
    }))
}

fn intrinsic_target() -> CausalQuery {
    CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::IntrinsicInfluence(
        IntrinsicInfluenceTarget {
            target: VariableId::from_raw(2),
            allocation: InfluenceAllocation::Exact,
            outer_draws: 40,
            inner_draws: 10,
            replicates: 3,
            fit_bootstrap_replicates: 0,
            level: 0.95,
            max_sources: 8,
            seed: 11,
            noise: NoiseDeclaration::Independent,
        },
    )))
}

fn bayesian(draws: usize) -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(draws))
}

fn change_study(rooted: bool, inference: InferenceMode, replicates: u32) -> Study {
    Study::tabular(two_period_table())
        .graph(two_node_dag())
        .query(change_target(rooted))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap()
}

#[test]
fn a_change_attribution_has_a_posterior_over_the_mechanisms() {
    on_large_stack("attribute-change-posterior", || {
        for rooted in [false, true] {
            let study = change_study(rooted, bayesian(24), 0);
            let ctx = ctx();
            let plan = study.compile(&ctx).unwrap();
            let lowered = plan.lowered().expect("compile attaches a lowered program");
            assert_eq!(lowered.route(), "GcmAttributeDag");
            let execution = CompiledExecution::run(&study, &ctx).unwrap();
            let result = execution.result();
            let posterior = result.posterior.as_ref().expect("a posterior over the mechanisms");
            assert_eq!(posterior.draws.n_draws, 24);
            // The total first, then one column per mechanism that changed.
            assert!(posterior.draws.schema.n_quantities() >= 2);
            let total = posterior.summaries.mean[0];
            assert!(
                (total - 5.0).abs() < 1.0,
                "the shifted intercept explains the change: {total}"
            );
            assert!(posterior.summaries.sd[0] > 0.0, "the refits differ from draw to draw");
            let components: f64 = posterior.summaries.mean[1..].iter().sum();
            assert!(
                (components - total).abs() < 0.05,
                "the components of one draw sum to its total, so their means do too"
            );
            assert!(
                result.diagnostics.iter().any(|d| d.code.as_ref() == "gcm.attribution.bayesian"),
                "the posterior says what it conditions on"
            );
            // The point answer is the fitted model's, and the answer slot is published.
            assert!(result.attribution().is_some());
        }
    });
}

#[test]
fn a_change_attribution_has_a_bootstrap_beside_its_posterior() {
    on_large_stack("attribute-change-bootstrap", || {
        let ctx = ctx();
        let frequentist =
            CompiledExecution::run(&change_study(false, InferenceMode::Frequentist, 40), &ctx)
                .unwrap();
        let result = frequentist.result();
        assert!(result.posterior.is_none(), "a frequentist answer carries no posterior");
        let estimate = result.estimate().as_effect().expect("the total change");
        let accounting = estimate.uncertainty.bootstrap;
        assert_eq!(
            accounting.replicates_ok.unwrap_or(0) + accounting.replicates_failed.unwrap_or(0),
            40
        );
        let se = accounting.se.expect("a bootstrap standard error");
        assert!(se > 0.0);
        assert!(
            (estimate.ate - 5.0).abs() < 1.0,
            "the fitted model's total change: {}",
            estimate.ate
        );
        assert!(
            result.diagnostics.iter().any(|d| d.code.as_ref() == "gcm.frequentist_bootstrap"),
            "the unlicensed bootstrap is disclosed"
        );
        // The posterior over the mechanisms and the bootstrap agree about the spread of the
        // total, up to the finite number of draws.
        let bayesian = CompiledExecution::run(&change_study(false, bayesian(60), 0), &ctx).unwrap();
        let posterior = bayesian.result().posterior.as_ref().unwrap();
        let ratio = posterior.summaries.sd[0] / se;
        assert!((0.4..2.5).contains(&ratio), "posterior sd over bootstrap se: {ratio}");
    });
}

#[test]
fn an_intrinsic_influence_has_a_posterior_over_the_mechanisms() {
    on_large_stack("attribute-intrinsic-posterior", || {
        let study = Study::tabular(chain_table())
            .graph(chain_dag())
            .query(intrinsic_target())
            .inference(bayesian(12))
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let ctx = ctx();
        let execution = CompiledExecution::run(&study, &ctx).unwrap();
        let result = execution.result();
        let posterior = result.posterior.as_ref().expect("a posterior over the mechanisms");
        assert_eq!(posterior.draws.n_draws, 12);
        assert_eq!(posterior.draws.schema.n_quantities(), 3, "one share per noise source");
        assert!(posterior.summaries.sd.iter().all(|sd| *sd > 0.0));
        let shares: f64 = posterior.summaries.mean.iter().sum();
        let total = match result.attribution().map(|payload| payload.outcome.clone()) {
            Some(antecedent::AttributionOutcome::IntrinsicInfluence(influence)) => {
                assert_eq!(influence.method.fit_resampling.as_str(), "dirichlet_row_weights");
                assert_eq!(influence.fit_draws.len(), 12);
                influence.total_variance
            }
            other => panic!("intrinsic influence returns its attribution: {other:?}"),
        };
        assert!(
            shares > 0.0 && shares < 2.0 * total,
            "the shares account for the target variance: {shares} against {total}"
        );
        assert!(
            result.diagnostics.iter().any(|d| d.code.as_ref() == "gcm.attribution.bayesian"),
            "the posterior says what it conditions on"
        );
    });
}

/// One target of each remaining kind on the two-period `x -> y` fixture, by name.
fn refit_targets() -> Vec<(&'static str, AttributeTarget)> {
    let change = || {
        ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: 100 },
            PopulationSelector::TimeRange { start: 100, end: 200 },
        )
    };
    vec![
        (
            "paths",
            AttributeTarget::Paths {
                sources: std::sync::Arc::from([VariableId::from_raw(0)]),
                outcome: VariableId::from_raw(1),
                max_paths: 8,
                max_len: 4,
                through: std::sync::Arc::from([]),
            },
        ),
        (
            "feature_relevance",
            AttributeTarget::FeatureRelevance {
                outcome: VariableId::from_raw(1),
                features: std::sync::Arc::from([VariableId::from_raw(0)]),
                delta: 1.0,
                n_samples: 200,
                max_features: 4,
            },
        ),
        (
            "anomaly",
            AttributeTarget::Anomaly(AnomalyAttributionQuery::new([VariableId::from_raw(1)], 200)),
        ),
        (
            "unit_change",
            AttributeTarget::UnitChange(
                UnitChangeQuery::new(VariableId::from_raw(1), 20)
                    .with_unit_rows((0..20).collect::<Vec<usize>>()),
            ),
        ),
        (
            "structure_change",
            AttributeTarget::StructureChange {
                change: change().with_components(AttributionComponents::Structure),
                comparison: std::sync::Arc::from([]),
                measure: ChangeMeasure::MeanDifference,
                n_samples: 200,
                seed: 5,
            },
        ),
        (
            "robust_change",
            AttributeTarget::DistributionChange {
                change: change(),
                estimator: ChangeEstimator::Robust { max_rows: 1000, n_samples: 200, seed: 5 },
            },
        ),
    ]
}

fn target_study(target: AttributeTarget, inference: InferenceMode, replicates: u32) -> Study {
    Study::tabular(two_period_table())
        .graph(two_node_dag())
        .query(CausalQuery::Attribute(AttributeQuery::new(target)))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap()
}

#[test]
fn every_refit_target_has_a_posterior_and_a_bootstrap() {
    on_large_stack("attribute-refit-targets", || {
        let ctx = ctx();
        for (name, target) in refit_targets() {
            let posterior =
                CompiledExecution::run(&target_study(target.clone(), bayesian(12), 0), &ctx)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            let result = posterior.result();
            let built = result.posterior.as_ref().unwrap_or_else(|| panic!("{name}: no posterior"));
            assert_eq!(built.draws.n_draws, 12, "{name}");
            assert!(built.summaries.mean.iter().all(|m| m.is_finite()), "{name}: finite means");
            assert!(built.summaries.sd.iter().any(|sd| *sd > 0.0), "{name}: the refits differ");
            assert!(
                result.diagnostics.iter().any(|d| d.code.as_ref() == "gcm.attribution.bayesian"),
                "{name}"
            );

            let bootstrap =
                CompiledExecution::run(&target_study(target, InferenceMode::Frequentist, 16), &ctx)
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
            let result = bootstrap.result();
            assert!(result.posterior.is_none(), "{name}");
            let disclosure = result
                .diagnostics
                .iter()
                .find(|d| d.code.as_ref() == "gcm.frequentist_bootstrap")
                .unwrap_or_else(|| panic!("{name}: the bootstrap is disclosed"));
            let ok: u32 = disclosure
                .fields
                .iter()
                .find(|(key, _)| key.as_ref() == "replicates_ok")
                .and_then(|(_, value)| value.parse().ok())
                .unwrap_or(0);
            assert!(ok >= 8, "{name}: replicates ok {ok}");
        }
    });
}

#[test]
fn mechanism_change_detection_has_no_posterior_and_says_why() {
    on_large_stack("attribute-mechanism-change", || {
        let detection = MechanismChangeQuery::new(
            [VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: 100 },
            PopulationSelector::TimeRange { start: 100, end: 200 },
            0.05,
            4,
        );
        let target = AttributeTarget::MechanismChange {
            query: detection,
            test: MechanismChangeTest::LikelihoodRatio,
        };
        let refused = Study::tabular(two_period_table())
            .graph(two_node_dag())
            .query(CausalQuery::Attribute(AttributeQuery::new(target.clone())))
            .inference(bayesian(12))
            .refute(RefuteSuite::None)
            .build()
            .expect_err("a test has no posterior over the mechanisms");
        let message = refused.to_string();
        assert!(message.contains("not a smooth functional"), "{message}");
        assert!(message.contains("Next attempt"), "{message}");
        // The Frequentist test runs as it always did.
        let run =
            CompiledExecution::run(&target_study(target, InferenceMode::Frequentist, 0), &ctx())
                .unwrap();
        assert!(run.result().posterior.is_none());
    });
}

/// `x_t = 0.5 x_{t-1} + e`, `y_t = intercept + 2 x_{t-1} + e'`, intercept 1 then 6.
fn shifted_series() -> TimeSeriesData {
    series_with_shift(5.0)
}

fn series_with_shift(shift: f64) -> TimeSeriesData {
    let mut stream = Stream(0x5EED_5EED);
    let n = 300;
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = 0.5 * x[t - 1] + stream.normal();
        let intercept = if t < n / 2 { 1.0 } else { 1.0 + shift };
        y[t] = intercept + 2.0 * x[t - 1] + stream.normal();
    }
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

fn temporal_chain() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let parent = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let child = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(parent, child).unwrap();
    graph
}

fn temporal_study(target: AttributeTarget, inference: InferenceMode, replicates: u32) -> Study {
    Study::series(shifted_series())
        .graph(temporal_chain())
        .query(CausalQuery::Attribute(AttributeQuery::new(target)))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap()
}

#[test]
fn a_temporal_dag_attribution_resamples_blocks_in_both_inference_modes() {
    on_large_stack("attribute-temporal", || {
        let ctx = ctx();
        let change = ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: 150 },
            PopulationSelector::TimeRange { start: 150, end: 300 },
        );
        let target = AttributeTarget::DistributionChange {
            change,
            estimator: ChangeEstimator::Sampled {
                measure: ChangeMeasure::MeanDifference,
                n_samples: 400,
                seed: 3,
            },
        };
        // Bayesian: block-weight draws of the lag-aligned unfolding.
        let study = temporal_study(target.clone(), bayesian(24), 0);
        let prepared = study.prepare(&ctx).unwrap();
        let result = prepared.estimate_series(&shifted_series(), &ctx).unwrap();
        let posterior = result.posterior.as_ref().expect("a block-weight posterior");
        assert_eq!(posterior.draws.n_draws, 24);
        assert!((posterior.summaries.mean[0] - 5.0).abs() < 1.5, "{}", posterior.summaries.mean[0]);
        assert!(posterior.summaries.sd[0] > 0.0);
        assert!(
            result
                .diagnostics
                .iter()
                .any(|d| d.code.as_ref() == "gcm.attribution.bayesian"
                    && d.message.contains("block")),
            "the block weights are disclosed"
        );
        // Frequentist: circular-block counts, a row-block interval.
        let study = temporal_study(target, InferenceMode::Frequentist, 49);
        let result = study.prepare(&ctx).unwrap().estimate_series(&shifted_series(), &ctx).unwrap();
        let estimate = result.estimate().as_effect().unwrap();
        let se = estimate.se_bootstrap().expect("a block-bootstrap standard error");
        assert!(se > 0.0 && se < 2.0, "se {se}");
        assert!((estimate.ate - 5.0).abs() < 5.0 * se + 0.5, "total {}", estimate.ate);
        assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "gcm.frequentist_bootstrap"
            && d.message.contains("circular-block")));
        // Intrinsic influence: block weights inside the engine.
        let influence = AttributeTarget::IntrinsicInfluence(IntrinsicInfluenceTarget {
            target: VariableId::from_raw(1),
            allocation: InfluenceAllocation::Exact,
            outer_draws: 40,
            inner_draws: 10,
            replicates: 3,
            fit_bootstrap_replicates: 0,
            level: 0.95,
            max_sources: 8,
            seed: 11,
            noise: NoiseDeclaration::Independent,
        });
        let study = temporal_study(influence, bayesian(12), 0);
        let result = study.prepare(&ctx).unwrap().estimate_series(&shifted_series(), &ctx).unwrap();
        let posterior = result.posterior.as_ref().expect("an intrinsic block-weight posterior");
        assert_eq!(posterior.draws.n_draws, 12);
        match result.attribution().map(|payload| payload.outcome.clone()) {
            Some(antecedent::AttributionOutcome::IntrinsicInfluence(influence)) => {
                assert_eq!(influence.method.fit_resampling.as_str(), "dirichlet_block_weights");
            }
            other => panic!("intrinsic influence returns its attribution: {other:?}"),
        }
    });
}

#[test]
fn standalone_temporal_queries_say_what_runs_and_what_does_not() {
    on_large_stack("standalone-temporal-queries", || {
        let ctx = ctx();
        // A unit change runs on the lag-aligned unfolding as the fitted model's point answer.
        let study = Study::series(shifted_series())
            .graph(temporal_chain())
            .query(CausalQuery::UnitChange(
                UnitChangeQuery::new(VariableId::from_raw(1), 20)
                    .with_unit_rows((0..20).collect::<Vec<usize>>()),
            ))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let result = study.prepare(&ctx).unwrap().estimate_series(&shifted_series(), &ctx).unwrap();
        assert!(result.posterior.is_none());
        // A mechanism-change test on one dependent series draws its null law by block
        // permutation of the residuals, and reads as a posterior probability of a change under
        // Bayesian inference and a declared prior.
        let detection = || {
            MechanismChangeQuery::new(
                [VariableId::from_raw(1)],
                PopulationSelector::TimeRange { start: 0, end: 150 },
                PopulationSelector::TimeRange { start: 150, end: 300 },
                0.05,
                4,
            )
        };
        let run =
            |series: TimeSeriesData, inference: InferenceMode, query: MechanismChangeQuery| {
                Study::series(series.clone())
                    .graph(temporal_chain())
                    .query(CausalQuery::MechanismChange(query))
                    .inference(inference)
                    .refute(RefuteSuite::None)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap()
                    .prepare(&ctx)
                    .unwrap()
                    .estimate_series(&series, &ctx)
            };
        let shifted = run(shifted_series(), InferenceMode::Frequentist, detection()).unwrap();
        let found = &shifted.mechanism_change().expect("detections")[0];
        assert!(found.changed, "the shifted intercept is detected: p = {}", found.p_value);
        assert!(found.method.contains("block_permutation"), "{}", found.method);
        assert!(found.n_permutations.is_some_and(|b| b >= 99));
        let null = run(series_with_shift(0.0), InferenceMode::Frequentist, detection()).unwrap();
        let quiet = &null.mechanism_change().unwrap()[0];
        assert!(quiet.p_value > 0.01, "no shift, no detection: p = {}", quiet.p_value);
        let bayes = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(8));
        let posterior =
            run(shifted_series(), bayes.clone(), detection().with_change_prior(0.5)).unwrap();
        let probability = posterior.mechanism_change().unwrap()[0].posterior_probability;
        assert!(probability.is_some_and(|p| p > 0.95), "{probability:?}");
        let calm =
            run(series_with_shift(0.0), bayes.clone(), detection().with_change_prior(0.5)).unwrap();
        assert!(calm.mechanism_change().unwrap()[0].posterior_probability.is_some_and(|p| p < 0.5));
        // The prior over the change indicator is declared, never assumed.
        assert!(run(shifted_series(), bayes, detection()).is_err());
    });
}

fn rollout_study(
    inference: InferenceMode,
    replicates: u32,
    schedule: Vec<SequencedIntervention>,
) -> Study {
    let target = SimulationTarget::TemporalRollout {
        schedule: InterventionSequence::new(schedule),
        n_rows: 400,
        n_steps: 4,
        window: 0,
        history: std::sync::Arc::from([]),
        seed: 9,
    };
    Study::series(series_with_shift(0.0))
        .graph(temporal_chain())
        .query(CausalQuery::Simulate(SimulateQuery::new(target)))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap()
}

#[test]
fn a_temporal_rollout_steps_the_fitted_dynamics_and_carries_block_refit_uncertainty() {
    on_large_stack("temporal-rollout", || {
        let ctx = ctx();
        let series = series_with_shift(0.0);
        let do_x = |level: f64| {
            vec![SequencedIntervention::new(
                Intervention::set(VariableId::from_raw(0), Value::f64(level)),
                TemporalPolicy::Sustained { from: 0, until: 3 },
            )]
        };
        let mean_y = |study: &Study, step: usize| {
            let result = study.prepare(&ctx).unwrap().estimate_series(&series, &ctx).unwrap();
            let payload = result.simulation().expect("a simulation payload");
            let antecedent::SimulationOutcome::Temporal(path) = &payload.outcome else {
                panic!("a rollout returns its path");
            };
            let column = path.column(step, 1).unwrap();
            column.iter().sum::<f64>() / column.len() as f64
        };
        // y_t = 1 + 2 x_{t-1} + noise: holding x at 1 from step 0 puts y at 3 from step 1 on,
        // through the fitted lag, which sampling the unfolded graph once could not show.
        let held = rollout_study(InferenceMode::Frequentist, 0, do_x(1.0));
        assert!((mean_y(&held, 2) - 3.0).abs() < 0.3, "{}", mean_y(&held, 2));
        let free = rollout_study(InferenceMode::Frequentist, 0, Vec::new());
        assert!((mean_y(&free, 2) - 1.0).abs() < 0.5, "{}", mean_y(&free, 2));
        // The parameter uncertainty of the dynamics: block-weight draws and block counts.
        let bayes = rollout_study(
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(12)),
            0,
            do_x(1.0),
        );
        let result = bayes.prepare(&ctx).unwrap().estimate_series(&series, &ctx).unwrap();
        let posterior = result.posterior.as_ref().expect("a posterior over the dynamics");
        assert_eq!(posterior.draws.n_draws, 12);
        assert_eq!(
            posterior.draws.schema.n_quantities(),
            16,
            "four steps by two variables, as means and as simulated-path draws"
        );
        assert!(posterior.summaries.sd.iter().any(|sd| *sd > 0.0));
        // The simulated-path columns are draws of the predictive law: refit uncertainty plus the
        // innovation noise (sd 1, rolled out under a fresh seed in every refit), so they spread
        // over about the innovation sd while the mean of the same step spreads over the refit
        // error only (about 0.1 on 300 rows).
        let (mean_at, path_at) = (2 * 2 + 1, 8 + 2 * 2 + 1);
        assert!((posterior.summaries.mean[path_at] - 3.0).abs() < 1.0);
        assert!(
            (0.6..1.6).contains(&posterior.summaries.sd[path_at])
                && posterior.summaries.sd[path_at] > 3.0 * posterior.summaries.sd[mean_at],
            "{} vs {}",
            posterior.summaries.sd[path_at],
            posterior.summaries.sd[mean_at]
        );
        let boot = rollout_study(InferenceMode::Frequentist, 12, do_x(1.0));
        let result = boot.prepare(&ctx).unwrap().estimate_series(&series, &ctx).unwrap();
        assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "gcm.frequentist_bootstrap"
            && d.message.contains("circular-block")));
    });
}

#[test]
fn every_simulation_target_carries_refit_uncertainty_in_both_modes() {
    on_large_stack("simulation-refits", || {
        let ctx = ctx();
        let do_t = || -> std::sync::Arc<[Intervention]> {
            std::sync::Arc::from([Intervention::set(VariableId::from_raw(1), Value::f64(1.0))])
        };
        let (z, t, y) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
        let _ = z;
        let targets = vec![
            (
                "draws",
                SimulationTarget::InterventionalSample {
                    interventions: do_t(),
                    draws: 200,
                    seed: 3,
                },
                3,
            ),
            (
                "policy_value",
                SimulationTarget::PolicyValue { interventions: do_t(), outcome: y, draws: 200 },
                3,
            ),
            (
                "unit_effects",
                SimulationTarget::UnitEffects {
                    treatment: t,
                    outcome: y,
                    active: 1.0,
                    control: 0.0,
                },
                1,
            ),
            (
                "counterfactual",
                SimulationTarget::Counterfactual {
                    interventions: do_t(),
                    outcomes: std::sync::Arc::from([y]),
                },
                1,
            ),
        ];
        for (name, target, quantities) in targets {
            let study = |inference: InferenceMode, replicates: u32| {
                Study::tabular(chain_table())
                    .graph(chain_dag())
                    .query(CausalQuery::Simulate(SimulateQuery::new(target.clone())))
                    .inference(inference)
                    .refute(RefuteSuite::None)
                    .bootstrap_replicates(replicates)
                    .build()
                    .unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            let bayes = CompiledExecution::run(&study(bayesian(10), 0), &ctx)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let posterior = bayes.result().posterior.as_ref().unwrap_or_else(|| panic!("{name}"));
            assert_eq!(posterior.draws.n_draws, 10, "{name}");
            assert!(posterior.draws.schema.n_quantities() >= quantities, "{name}");
            assert!(posterior.summaries.mean.iter().all(|m| m.is_finite()), "{name}");
            let boot = CompiledExecution::run(&study(InferenceMode::Frequentist, 10), &ctx)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(
                boot.result()
                    .diagnostics
                    .iter()
                    .any(|d| d.code.as_ref() == "gcm.frequentist_bootstrap"),
                "{name}"
            );
        }
    });
}

#[test]
fn an_attribute_mechanism_change_has_a_block_null_for_every_statistic_and_a_declared_posterior() {
    on_large_stack("attribute-mechanism-change-temporal", || {
        let ctx = ctx();
        let detection = || {
            MechanismChangeQuery::new(
                [VariableId::from_raw(1)],
                PopulationSelector::TimeRange { start: 0, end: 150 },
                PopulationSelector::TimeRange { start: 150, end: 300 },
                0.05,
                4,
            )
        };
        let run = |series: TimeSeriesData,
                   inference: InferenceMode,
                   query: MechanismChangeQuery,
                   test: MechanismChangeTest|
         -> Result<antecedent::StudyResult, String> {
            Study::series(series.clone())
                .graph(temporal_chain())
                .query(CausalQuery::Attribute(AttributeQuery::new(
                    AttributeTarget::MechanismChange { query, test },
                )))
                .inference(inference)
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .map_err(|error| error.to_string())?
                .prepare(&ctx)
                .map_err(|error| error.to_string())?
                .estimate_series(&series, &ctx)
                .map_err(|error| error.to_string())
        };
        let detections = |result: &antecedent::StudyResult| match result
            .attribution()
            .map(|payload| payload.outcome.clone())
        {
            Some(antecedent::AttributionOutcome::MechanismChange(found)) => found,
            other => panic!("a mechanism-change attribution returns detections: {other:?}"),
        };
        for test in [
            MechanismChangeTest::LikelihoodRatio,
            MechanismChangeTest::MeanDifference,
            MechanismChangeTest::KernelTwoSample,
            MechanismChangeTest::ClassifierTwoSample,
            MechanismChangeTest::ChangePoint,
        ] {
            let shifted =
                run(shifted_series(), InferenceMode::Frequentist, detection(), test).unwrap();
            let found = &detections(&shifted)[0];
            assert!(found.method.contains("block_permutation"), "{test:?}: {}", found.method);
            assert!(found.p_value < 0.05, "{test:?}: shift not detected, p = {}", found.p_value);
        }
        // The Bayesian reading is the posterior probability of a change under the declared prior.
        let bayes = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(8));
        let posterior = run(
            shifted_series(),
            bayes.clone(),
            detection().with_change_prior(0.5),
            MechanismChangeTest::LikelihoodRatio,
        )
        .unwrap();
        assert!(detections(&posterior)[0].posterior_probability.is_some_and(|p| p > 0.95));
        let calm = run(
            series_with_shift(0.0),
            bayes.clone(),
            detection().with_change_prior(0.5),
            MechanismChangeTest::LikelihoodRatio,
        )
        .unwrap();
        assert!(detections(&calm)[0].posterior_probability.is_some_and(|p| p < 0.5));
        // No prior is assumed.
        assert!(
            run(shifted_series(), bayes, detection(), MechanismChangeTest::LikelihoodRatio)
                .is_err()
        );
    });
}

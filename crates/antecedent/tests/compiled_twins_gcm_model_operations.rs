//! The direct fitted-model operations of `antecedent::gcm` and their compiled replacements give
//! the same numbers.
//!
//! `PreparedStudy::model_operation` fits the study's mechanism program (here the standard one,
//! which selects the mechanisms `fit_gcm` selects) on the study's own DAG and table and calls the
//! same kernel with the same inputs:
//!
//! * `arrow_strengths` and `ModelOperation::ArrowStrengths`.
//! * `population_do_contrast` and `ModelOperation::PopulationDoContrast`. The operation draws
//!   both arms from the attribution stream of a generator seeded with the operation's seed; the
//!   direct call draws them from its context's generator. A context seeded with the operation's
//!   seed gives the same numbers, whatever seed the study itself runs under.
//! * `KdeDoSampler::sample`, `McmcDoSampler::sample` and `WeightingDoSampler::estimate` and
//!   `ModelOperation::DoSample`: the kernel-density and Metropolis-Hastings samplers start from
//!   `CausalRng::from_seed(seed)` on both sides; the weighting sampler draws nothing.
//! * `ModelEvaluator::evaluate` and `ModelOperation::ModelEvaluation`.
//! * `nested_counterfactual`, `nested_counterfactual_with_exo` and `nested_hard_counterfactual`
//!   and `ModelOperation::NestedCounterfactual`: abduction is deterministic on a complete table,
//!   so all three are one number.
//! * `sample_interventional` and an `InterventionalSample` target, `sample_observational` and an
//!   `InterventionalSample` target with no intervention: both start from
//!   `CausalRng::from_seed(seed)`.
//! * `simultaneous_hard_counterfactual` and the counterfactual `SimulateQuery` declared with
//!   `SimulateQuery::simultaneous_world`, which refuses overlapping sets when it is
//!   declared, as the direct entry refuses them when it runs.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    deprecated,
    reason = "these tests call the direct operations the compiled routes are compared with"
)]
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "bit-identical equality is the property under test; columns come from small indices"
)]
use std::sync::Arc;

use antecedent::analysis::model_operation::ModelOperationAnswer;
use antecedent::gcm::{
    AbductionMissingPolicy, CounterfactualEngine, DoSampleResult, KdeDoSampler, McmcDoSampler,
    MechanismWorkspace, ModelEvaluator, WeightingDoSampler, arrow_strengths, fit_gcm,
    nested_counterfactual, nested_counterfactual_with_exo, nested_hard_counterfactual,
    population_do_contrast, sample_interventional, sample_observational,
    simultaneous_hard_counterfactual,
};
use antecedent::{CompiledExecution, InferenceMode, RefuteSuite, SimulationOutcome, Study};
use antecedent_core::{
    ArrowStrengthsTarget, CausalQuery, CausalRng, DoSamplerMethod, DoSamplerTarget,
    ExecutionContext, Intervention, ModelEvaluationTarget, ModelOperation,
    NestedCounterfactualTarget, PopulationDoContrastTarget, QueryError, SimulateQuery,
    SimulationTarget, Value, VariableId,
};
use antecedent_counterfactual::NestedSpec;
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};

const ROWS: usize = 300;
const SEED: u64 = 0x5EED_0D0C;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(SEED)
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}

fn opt_bits(value: Option<f64>) -> Option<u64> {
    value.map(f64::to_bits)
}

fn hard(variable: u32, level: f64) -> Intervention {
    Intervention::set(v(variable), Value::f64(level))
}

fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    antecedent::support::caller_thread()
        .stack_size(32 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

/// `x -> y`, `x -> z`, `y -> z`, linear with bounded wiggles. Variables `0 = x`, `1 = y`,
/// `2 = z`.
fn linear() -> (TabularData, Dag) {
    let x: Vec<f64> =
        (0..ROWS).map(|i| (i % 17) as f64 * 0.2 - 1.6 + 0.2 * (i as f64 * 1.3).cos()).collect();
    let y: Vec<f64> = (0..ROWS).map(|i| 1.0 + 2.0 * x[i] + 0.4 * (i as f64 * 0.37).sin()).collect();
    let z: Vec<f64> =
        (0..ROWS).map(|i| -0.5 + 0.7 * x[i] + 1.5 * y[i] + 0.4 * (i as f64 * 0.91).cos()).collect();
    let data = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    (data, graph)
}

fn study(data: &TabularData, graph: &Dag, query: CausalQuery) -> Study {
    Study::tabular(data.clone())
        .graph(graph.clone())
        .query(query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

/// Run `operation` on the fitted model of a study prepared on `data` and `graph`, under `ctx`.
/// The study's own question (an empty interventional sample) is not read by the operation.
fn operate(
    data: &TabularData,
    graph: &Dag,
    operation: &ModelOperation,
    ctx: &ExecutionContext,
) -> ModelOperationAnswer {
    let carrier =
        CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::InterventionalSample {
            interventions: Arc::from([]),
            draws: 1,
            seed: 0,
        }));
    let prepared = study(data, graph, carrier).prepare(ctx).unwrap();
    let compiled = prepared.model_operation(operation, ctx).unwrap();
    let family =
        if operation.is_attribution() { "FittedModelAttribution" } else { "FittedModelSimulation" };
    assert_eq!(compiled.lowered().route(), family);
    compiled.into_answer()
}

/// Run `query` through the compiled `Simulate` route under `ctx`.
fn simulate(
    data: &TabularData,
    graph: &Dag,
    query: SimulateQuery,
    ctx: &ExecutionContext,
) -> SimulationOutcome {
    let study = study(data, graph, CausalQuery::Simulate(query));
    let lowered = study.compile(ctx).unwrap();
    assert_eq!(lowered.lowered().expect("a lowered program").route(), "GcmSimulateDag");
    let execution = CompiledExecution::run(&study, ctx).unwrap();
    execution.result().simulation().map(|payload| payload.outcome.clone()).expect("a payload")
}

fn do_sample(data: &TabularData, graph: &Dag, target: DoSamplerTarget) -> DoSampleResult {
    match operate(data, graph, &ModelOperation::DoSample(target), &ctx()) {
        ModelOperationAnswer::DoSample(result) => result,
        other => panic!("a do-sampler target returns its sample: {other:?}"),
    }
}

fn assert_do_sample_eq(compiled: &DoSampleResult, direct: &DoSampleResult) {
    assert!(!direct.values.is_empty(), "the direct sampler drew");
    assert_eq!(bits(&compiled.values), bits(&direct.values));
    assert_eq!(bits(&compiled.weights), bits(&direct.weights));
    assert_eq!(compiled.method, direct.method);
    assert_eq!(compiled.notes, direct.notes);
    assert_eq!(opt_bits(compiled.accept_rate), opt_bits(direct.accept_rate));
    assert_eq!(opt_bits(compiled.bandwidth), opt_bits(direct.bandwidth));
    assert_eq!(opt_bits(compiled.ess), opt_bits(direct.ess));
}

#[test]
fn arrow_strengths_equal_the_compiled_arrow_strengths_route() {
    on_large_stack(|| {
        let (data, graph) = linear();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let direct = arrow_strengths(&fitted.model).unwrap();
        let ModelOperationAnswer::ArrowStrengths(compiled) =
            operate(&data, &graph, &ModelOperation::ArrowStrengths(ArrowStrengthsTarget), &ctx())
        else {
            panic!("an arrow-strengths target returns the strengths");
        };
        assert_eq!(compiled.len(), direct.len());
        assert_eq!(direct.len(), 3, "one strength per edge");
        for (compiled, direct) in compiled.iter().zip(&direct) {
            assert_eq!((compiled.parent, compiled.child), (direct.parent, direct.child));
            assert_eq!(compiled.strength.to_bits(), direct.strength.to_bits());
            assert_eq!(
                compiled.variance_contribution.to_bits(),
                direct.variance_contribution.to_bits()
            );
            assert_eq!(compiled.coefficient.to_bits(), direct.coefficient.to_bits());
        }
    });
}

#[test]
fn population_do_contrast_draws_from_the_target_seed_like_a_context_seeded_with_it() {
    on_large_stack(|| {
        let (data, graph) = linear();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let target_seed = 41;
        let direct = population_do_contrast(
            &fitted.model,
            &data,
            v(0),
            v(2),
            0.5,
            data.row_count(),
            &ExecutionContext::for_tests(target_seed),
        )
        .unwrap();
        let target = PopulationDoContrastTarget {
            parent: v(0),
            child: v(2),
            delta: 0.5,
            max_units: None,
            seed: target_seed,
        };
        // The study runs under another seed: the draws follow the target's.
        let ModelOperationAnswer::PopulationDoContrast(compiled) = operate(
            &data,
            &graph,
            &ModelOperation::PopulationDoContrast(target),
            &ExecutionContext::for_tests(target_seed ^ 0xFFFF),
        ) else {
            panic!("a do-contrast target returns the contrast");
        };
        assert!(direct.contrast.is_finite());
        assert_eq!(compiled.contrast.to_bits(), direct.contrast.to_bits());
        assert_eq!(compiled.stderr.to_bits(), direct.stderr.to_bits());
    });
}

#[test]
fn kde_do_sampler_equals_the_compiled_do_sample_route() {
    on_large_stack(|| {
        let (data, graph) = linear();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let interventions: Arc<[Intervention]> = Arc::from([hard(0, 0.5)]);
        let mut sampler = KdeDoSampler::new(v(2));
        sampler.bandwidth = Some(0.3);
        let mut rng = CausalRng::from_seed(7);
        let mut ws = MechanismWorkspace::default();
        let direct =
            sampler.sample(&fitted.model, &interventions, 200, &mut rng, &mut ws, &ctx()).unwrap();
        let compiled = do_sample(
            &data,
            &graph,
            DoSamplerTarget {
                method: DoSamplerMethod::Kde { bandwidth: Some(0.3) },
                outcome: v(2),
                interventions,
                draws: 200,
                seed: 7,
            },
        );
        assert_do_sample_eq(&compiled, &direct);
    });
}

#[test]
fn mcmc_do_sampler_equals_the_compiled_do_sample_route() {
    on_large_stack(|| {
        let (data, graph) = linear();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let interventions: Arc<[Intervention]> = Arc::from([hard(0, 0.5)]);
        let mut sampler = McmcDoSampler::new(v(2));
        sampler.burn_in = 50;
        sampler.thin = 2;
        let mut rng = CausalRng::from_seed(11);
        let mut ws = MechanismWorkspace::default();
        let direct =
            sampler.sample(&fitted.model, &interventions, 150, &mut rng, &mut ws, &ctx()).unwrap();
        let compiled = do_sample(
            &data,
            &graph,
            DoSamplerTarget {
                method: DoSamplerMethod::Mcmc {
                    proposal_sd: None,
                    burn_in: Some(50),
                    thin: Some(2),
                },
                outcome: v(2),
                interventions,
                draws: 150,
                seed: 11,
            },
        );
        assert_do_sample_eq(&compiled, &direct);
    });
}

#[test]
fn weighting_do_sampler_equals_the_compiled_do_sample_route() {
    on_large_stack(|| {
        let (data, graph) = linear();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        // `y` has a parent, so the sampler reweights the observed rows toward do(y = 1).
        let direct = WeightingDoSampler::new(v(1), v(2))
            .estimate(&fitted.model, &data, 1.0, &ctx())
            .unwrap();
        let compiled = do_sample(
            &data,
            &graph,
            DoSamplerTarget {
                method: DoSamplerMethod::Weighting { root_match_tolerance: None },
                outcome: v(2),
                interventions: Arc::from([hard(1, 1.0)]),
                draws: 1,
                seed: 0,
            },
        );
        assert_do_sample_eq(&compiled, &direct);
    });
}

#[test]
fn model_evaluator_equals_the_compiled_model_evaluation_route() {
    on_large_stack(|| {
        let (data, graph) = linear();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let direct = ModelEvaluator { alpha: 0.05, n_permutations: 8, seed: 5 }
            .evaluate(&fitted.model, &data, &ctx())
            .unwrap();
        let operation = ModelOperation::ModelEvaluation(ModelEvaluationTarget {
            alpha: 0.05,
            n_permutations: 8,
            seed: 5,
        });
        let ModelOperationAnswer::ModelEvaluation(compiled) =
            operate(&data, &graph, &operation, &ctx())
        else {
            panic!("a model evaluation returns its report");
        };
        assert_eq!(compiled.in_sample_loglik.to_bits(), direct.in_sample_loglik.to_bits());
        assert_eq!(compiled.zero_probability_rows, direct.zero_probability_rows);
        assert_eq!(compiled.mean_abs_residual.to_bits(), direct.mean_abs_residual.to_bits());
        assert_eq!(bits(&compiled.residual_independence_p), bits(&direct.residual_independence_p));
        assert_eq!(bits(&compiled.local_markov_p), bits(&direct.local_markov_p));
        assert_eq!(compiled.permutation_loglik.to_bits(), direct.permutation_loglik.to_bits());
        assert_eq!(compiled.falsified, direct.falsified);
        assert_eq!(compiled.alpha.to_bits(), direct.alpha.to_bits());
        assert_eq!(compiled.notes, direct.notes);
    });
}

fn sample_route(data: &TabularData, graph: &Dag, interventions: Arc<[Intervention]>) -> Vec<f64> {
    let query = SimulateQuery::new(SimulationTarget::InterventionalSample {
        interventions,
        draws: 64,
        seed: 13,
    });
    match simulate(data, graph, query, &ctx()) {
        SimulationOutcome::Sample(batch) => batch.values.to_vec(),
        other => panic!("an interventional sample returns its draws: {other:?}"),
    }
}

#[test]
fn sample_interventional_equals_the_compiled_interventional_sample() {
    on_large_stack(|| {
        let (data, graph) = linear();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let interventions: Arc<[Intervention]> = Arc::from([hard(1, 0.25)]);
        let mut rng = CausalRng::from_seed(13);
        let mut ws = MechanismWorkspace::default();
        let direct =
            sample_interventional(&fitted.model, &interventions, 64, &mut rng, &mut ws, &ctx())
                .unwrap();
        assert_eq!(bits(&sample_route(&data, &graph, interventions)), bits(&direct.values));
    });
}

#[test]
fn sample_observational_equals_the_compiled_sample_with_no_intervention() {
    on_large_stack(|| {
        let (data, graph) = linear();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let mut rng = CausalRng::from_seed(13);
        let mut ws = MechanismWorkspace::default();
        let direct = sample_observational(&fitted.model, 64, &mut rng, &mut ws, &ctx()).unwrap();
        assert_eq!(bits(&sample_route(&data, &graph, Arc::from([]))), bits(&direct.values));
    });
}

#[test]
fn simultaneous_counterfactual_declaration_equals_the_direct_engine_and_refuses_overlap() {
    on_large_stack(|| {
        let (data, graph) = linear();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let outer = [hard(0, 1.0)];
        let inner = [hard(1, 0.5)];
        let engine = CounterfactualEngine::new(fitted.model.clone());
        let mut ws = MechanismWorkspace::default();
        let direct =
            simultaneous_hard_counterfactual(&engine, &data, &outer, &inner, v(2), &mut ws, &ctx())
                .unwrap();
        let query = SimulateQuery::simultaneous_world(&outer, &inner, vec![v(2)]).unwrap();
        let SimulationOutcome::Counterfactual(values) = simulate(&data, &graph, query, &ctx())
        else {
            panic!("a counterfactual returns its values");
        };
        let compiled = values.outcome_mean(0, DenseNodeId::from_raw(2));
        assert_eq!(compiled.n_used, compiled.n_units, "every unit enters the mean");
        assert_eq!(compiled.mean.to_bits(), direct.to_bits());

        // Overlapping sets: the direct engine refuses when it runs, the declaration refuses
        // before any study is built.
        let overlapping = [hard(0, 0.0)];
        assert!(
            simultaneous_hard_counterfactual(
                &engine,
                &data,
                &outer,
                &overlapping,
                v(2),
                &mut ws,
                &ctx()
            )
            .is_err()
        );
        assert!(matches!(
            SimulateQuery::simultaneous_world(&outer, &overlapping, vec![v(2)]),
            Err(QueryError::InvalidSimulate(_))
        ));
    });
}

#[test]
fn the_nested_counterfactual_entries_equal_the_compiled_nested_operation() {
    on_large_stack(|| {
        let (data, graph) = linear();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        // Y_{x = 1, M_{x = 0}} with M = y: the mediation contrast of x on z through y.
        let outer = [hard(0, 0.0)];
        let inner = [hard(0, 1.0)];
        let frozen = [v(1)];
        let spec = NestedSpec { outer: &outer, inner: &inner, frozen: &frozen, outcome: v(2) };
        let engine = CounterfactualEngine::new(fitted.model.clone());
        let mut ws = MechanismWorkspace::default();
        let direct = nested_counterfactual(&engine, &data, &spec, &mut ws, &ctx()).unwrap();
        let hard_alias =
            nested_hard_counterfactual(&engine, &data, &spec, &mut ws, &ctx()).unwrap();
        let exo = engine.abduct(&data, AbductionMissingPolicy::Error, &ctx()).unwrap();
        let with_exo =
            nested_counterfactual_with_exo(&engine, &exo, &spec, &mut ws, &ctx()).unwrap();
        assert!(direct.is_finite());
        assert_eq!(hard_alias.to_bits(), direct.to_bits());
        assert_eq!(with_exo.to_bits(), direct.to_bits());

        let operation = ModelOperation::NestedCounterfactual(NestedCounterfactualTarget {
            outer: Arc::from(outer),
            inner: Arc::from(inner),
            frozen: Arc::from(frozen),
            outcome: v(2),
        });
        let ModelOperationAnswer::NestedCounterfactual(compiled) =
            operate(&data, &graph, &operation, &ctx())
        else {
            panic!("a nested counterfactual returns its mean");
        };
        assert_eq!(compiled.to_bits(), direct.to_bits());
    });
}

#[test]
fn a_nested_operation_that_freezes_its_outcome_is_refused_before_it_runs() {
    on_large_stack(|| {
        let (data, graph) = linear();
        let operation = ModelOperation::NestedCounterfactual(NestedCounterfactualTarget {
            outer: Arc::from([hard(0, 0.0)]),
            inner: Arc::from([hard(0, 1.0)]),
            frozen: Arc::from([v(2)]),
            outcome: v(2),
        });
        let carrier =
            CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::InterventionalSample {
                interventions: Arc::from([]),
                draws: 1,
                seed: 0,
            }));
        let prepared = study(&data, &graph, carrier).prepare(&ctx()).unwrap();
        assert!(prepared.model_operation(&operation, &ctx()).is_err());
    });
}

/// A compiled interventional sample seals with a contract whose claim verifies against the
/// sealed body: its calibration basis names the off-axis `Simulate` query the contract targets.
#[test]
fn a_sealed_interventional_sample_verifies_against_its_body() {
    on_large_stack(|| {
        let (data, graph) = linear();
        let ctx = ctx();
        let worlds: [Arc<[Intervention]>; 2] = [Arc::from([]), Arc::from([hard(1, 0.25)])];
        for interventions in worlds {
            let query =
                CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::InterventionalSample {
                    interventions,
                    draws: 64,
                    seed: 13,
                }));
            let prepared = study(&data, &graph, query).prepare(&ctx).unwrap();
            let result = prepared.estimate(&data, &ctx).unwrap();
            let bytes = prepared.encode_contracted_result(&result, "simulation", &ctx).unwrap();
            let retained = antecedent_io::RetainedArtifactIdentity::of_bytes(&bytes);
            let consumed = antecedent_io::replay_analysis_result_expecting(&bytes, &retained)
                .expect("the sealed contract verifies against the sealed body");
            let contract = consumed.contract.expect("the sealed artifact carries its contract");
            let basis = contract.claim.as_ref().unwrap().calibration.basis.as_ref();
            let basis = basis.expect("the claim carries the calibration basis it was matched by");
            assert_eq!(basis.key.query, "Simulate");
        }
    });
}

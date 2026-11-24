//! The chain that joins separately estimated outcomes is compiled before anything is fitted.
//!
//! Everything about the chain except the posterior moments is known at compile: its sources,
//! stages, factors, alignment, rows, seed and declared cross-study blocks. The compile lowers
//! the chain with its family table deferred, so the plan carries the chain's program identity
//! and the table's work budget is checked before any study is fitted; execution binds each
//! study's posterior moments to the table, draws it and runs the compiled program.
//!
//! * The program the receipt names is the one compiled; the fitted draws never reach it, and
//!   another seed or draw count is another program.
//! * The law drawn through the bound table is, bit for bit, the law the family table draws for
//!   the same moments and declaration when the chain is compiled only after fitting.
//! * A deferred program is never executed, sealed or exported as it is; a binding of the wrong
//!   shape or one the table cannot draw is refused; cancellation is honoured.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::native_claims::NativeDecisionSource;
use antecedent::analysis::preference_decision::bridge::chain_joint_law;
use antecedent::analysis::preference_decision::{
    AcceptedPreferences, ActionInput, ActionKind, Attribute, CompiledPreferenceDecision,
    CrossStudyCovariance, DecisionCriterion, OutcomeDependence, PreferenceAction,
    PreferenceContract, PreferenceContractParts, PreferenceDecisionExecution,
    PreferenceEvaluationOptions, PreferenceExpr, PreferenceParameter, PreferencePolicy,
    PreferenceSet, RowProvenance, SharedEvidence, StructuralPolicy, ValueModel,
    compile_preference_decision_across_studies, compile_preference_decision_with_shared_evidence,
    shape,
};
use antecedent::analysis::propagation_chain::{
    ChainCoefficientWire, ChainOpWire, ChainReading, ChainRefusal, ChainStageWire,
    ChainTableBinding, DeferredCrossBlock, FamilyDependence, FamilyEstimate, FamilyEstimateDecl,
    LawMeaning, PropagationChain, PropagationChainDraftWire, SharedFactorKind, StageKind,
    compile_deferred_propagation_chain, compile_propagation_chain, consume_propagation_chain,
    deferred_family_table, family_table,
};
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::distribution_artifact::DistributionArtifact;
use antecedent_io::propagation_chain_record::{ChainJointOutputWire, PropagationChainError};

use antecedent_core::reason_details::propagation_chain::DEFERRED_SOURCE_UNBOUND;

fn options() -> PreferenceEvaluationOptions {
    PreferenceEvaluationOptions::default()
}

fn wobble(i: u32, rate: f64) -> f64 {
    (f64::from(i) * rate).sin()
}

fn quantity(variable: &str, units: &str, dose: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: "patients".into(),
        regime_id: format!("do(t={dose})"),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

/// `benefit - harm_rate * harm - excess_weight * max(harm - cap, 0)`.
fn kinked(cap: f64) -> ValueModel {
    ValueModel {
        attributes: vec![
            Attribute { id: "benefit".into(), units: "points".into() },
            Attribute { id: "harm".into(), units: "events".into() },
        ],
        preferences: PreferenceSet::new(
            vec![
                PreferenceParameter {
                    id: "harm_rate".into(),
                    units: "points/events".into(),
                    lower: 0.1,
                    upper: 0.3,
                    assumption_id: "assumption.price_of_harm".into(),
                },
                PreferenceParameter {
                    id: "excess_weight".into(),
                    units: "points/events".into(),
                    lower: 0.0,
                    upper: 2.0,
                    assumption_id: "assumption.weight_of_excess_harm".into(),
                },
            ],
            vec![],
        ),
        value: PreferenceExpr::difference(
            PreferenceExpr::difference(
                PreferenceExpr::Attribute(0),
                PreferenceExpr::product(PreferenceExpr::Parameter(0), PreferenceExpr::Attribute(1)),
            ),
            PreferenceExpr::product(PreferenceExpr::Parameter(1), shape::excess(1, cap)),
        ),
        utility_units: "points".into(),
    }
}

fn contract(cap: f64) -> PreferenceContract {
    let action = |id: &str, dose: &str| PreferenceAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![
            ActionInput::Quantity(quantity("b", "points", dose)),
            ActionInput::Quantity(quantity("h", "events", dose)),
        ],
    };
    AcceptedPreferences::declared(kinked(cap)).unwrap().contract(PreferenceContractParts {
        actions: vec![action("high", "1.5"), action("low", "0.5")],
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "patients".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    })
}

fn dose_table(outcome: &str, rows: u32, (intercept, slope): (f64, f64), rate: f64) -> TabularData {
    let t: Vec<f64> = (0..rows).map(|i| 2.0 * f64::from(i) / f64::from(rows - 1)).collect();
    let y: Vec<f64> = (0..rows)
        .zip(&t)
        .map(|(i, dose)| intercept + slope * dose + 0.6 * wobble(i, rate))
        .collect();
    TabularData::from_f64_columns([("t", t.as_slice()), (outcome, y.as_slice())]).unwrap()
}

fn shared_table() -> TabularData {
    let rows = 200_u32;
    let t: Vec<f64> = (0..rows).map(|i| 2.0 * f64::from(i) / f64::from(rows - 1)).collect();
    let b: Vec<f64> =
        (0..rows).zip(&t).map(|(i, t)| 2.0 + 1.2 * t + 0.6 * wobble(i, 1.7)).collect();
    let h: Vec<f64> =
        (0..rows).zip(&t).map(|(i, t)| 0.5 + 0.5 * t + 0.6 * wobble(i, 2.9)).collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("b", b.as_slice()), ("h", h.as_slice())])
        .unwrap()
}

fn dose_study(data: &TabularData, variables: u32, outcome: u32) -> Study {
    let mut graph = Dag::with_variables(variables);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(outcome)).unwrap();
    Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(outcome),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.5, 1.5])),
            ),
        })))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        // A study of a whole table declares that table as its rows' source.
        .row_provenance(RowProvenance::of_table(data).unwrap())
        .build()
        .unwrap()
}

fn separate_studies() -> [Study; 2] {
    [
        dose_study(&dose_table("b", 220, (2.0, 1.2), 1.7), 2, 1),
        dose_study(&dose_table("h", 260, (0.5, 0.5), 2.9), 2, 1),
    ]
}

fn distinct(draws: usize, seed: u64) -> OutcomeDependence {
    OutcomeDependence::DistinctEvidence { draws, seed }
}

fn column(law: &DistributionArtifact, j: usize) -> Vec<f64> {
    let [n, width] = law.shape();
    (0..n).map(|draw| law.draws()[draw * width + j]).collect()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / f64::from(u32::try_from(values.len()).unwrap())
}

/// Sample covariance (divisor `n - 1`) of two aligned columns of draws: the convention the
/// binding of a study's posterior moments reads.
fn covariance(x: &[f64], y: &[f64]) -> f64 {
    let (mx, my) = (mean(x), mean(y));
    x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>()
        / f64::from(u32::try_from(x.len() - 1).unwrap())
}

/// The law the chain draws when it is declared and compiled only once the studies are fitted:
/// a family table matched to each study's posterior moments and the declared cross blocks,
/// drafted, compiled and executed after fitting.
fn compiled_after_fitting(
    compiled: &CompiledPreferenceDecision,
    executed: &PreferenceDecisionExecution,
    ctx: &ExecutionContext,
) -> DistributionArtifact {
    let fact = compiled.joint_law().expect("a declared joint law").fact();
    let mut estimates = Vec::new();
    let mut columns: Vec<(usize, Vec<f64>)> = Vec::new();
    let mut offsets = Vec::new();
    let mut outputs = Vec::new();
    for (study, input) in executed.inputs.iter().enumerate() {
        let NativeDecisionSource::JointLaw(law) = &input.source else {
            panic!("each study's posterior draws");
        };
        offsets.push(columns.len());
        for j in 0..law.shape()[1] {
            let values = column(law, j);
            let stage = format!("study_{study}_column_{j}");
            let factor = fact.inputs()[columns.len()].factor();
            estimates.push(
                FamilyEstimate::new(
                    &format!("{stage}_source"),
                    mean(&values),
                    covariance(&values, &values).sqrt(),
                    LawMeaning::Posterior,
                )
                .with_factor(SharedFactorKind::Row, factor),
            );
            outputs.push(ChainJointOutputWire { stage, quantity: law.quantities()[j].clone() });
            columns.push((study, values));
        }
    }
    let k = columns.len();
    let mut declared = vec![0.0; k * k];
    for (i, (left, x)) in columns.iter().enumerate() {
        for (j, (right, y)) in columns.iter().enumerate() {
            if left == right {
                declared[i * k + j] = covariance(x, y);
            }
        }
    }
    for block in compiled.cross_study_covariance() {
        let (first, second) = (offsets[block.first], offsets[block.second]);
        let rows = columns.iter().filter(|(study, _)| *study == block.first).count();
        let cols = columns.iter().filter(|(study, _)| *study == block.second).count();
        for p in 0..rows {
            for q in 0..cols {
                let term = block.covariance[p * cols + q];
                declared[(first + p) * k + second + q] = term;
                declared[(second + q) * k + first + p] = term;
            }
        }
    }
    let table = family_table(
        compiled.bound_identity(),
        &estimates,
        &FamilyDependence::Covariance(declared),
        None,
        fact.draws(),
        fact.seed(),
    )
    .expect("the family table");
    let stages = outputs
        .iter()
        .map(|output| {
            ChainStageWire::new(
                &output.stage,
                StageKind::DerivedQuantity,
                &[format!("{}_source", output.stage).as_str()],
                ChainOpWire::affine(
                    ChainCoefficientWire::fixed(1.0),
                    ChainCoefficientWire::fixed(0.0),
                ),
            )
        })
        .collect();
    let (output, primary) = (outputs[0].stage.clone(), outputs[0].quantity.clone());
    let draft = table
        .draft(ChainReading::Posterior, &output, stages)
        .with_output_quantity(primary)
        .with_joint_outputs(outputs);
    let chain = compile_propagation_chain(draft, ctx)
        .and_then(|plan| plan.execute(ctx))
        .expect("the chain compiled after fitting");
    chain_joint_law(&chain).expect("a joint law")
}

#[test]
fn the_chain_is_compiled_before_anything_is_fitted_and_is_the_program_executed() {
    let ctx = ExecutionContext::for_tests(53);
    let [benefit, harm] = separate_studies();
    let compiled = compile_preference_decision_across_studies(
        &[&benefit, &harm],
        &contract(1.2),
        distinct(2000, 7),
        &ctx,
    )
    .unwrap();
    let joint = compiled.joint_law().expect("a declared joint law");
    let program = joint.program();
    assert!(program.is_deferred(), "the table is bound at execution");
    assert_eq!(joint.fact().program(), program.identity());
    let chain_view = compiled.chain_plan_view().expect("the chain's plan view");
    assert_eq!(chain_view.identity(), program.identity());
    assert_eq!(chain_view.to_json()["evidence"]["chain"]["label"], joint.fact().label());
    for view in compiled.plan_views() {
        let json = view.to_json();
        let stated = &json["evidence"]["decision"]["preference"]["joint_law"];
        assert_eq!(stated["program"], program.identity().to_hex());
    }

    // The fitted draws never reach the compiled program: an execution under another seed draws
    // other posteriors and runs the same program.
    let executed = compiled.execute(None, options(), &ctx).unwrap();
    assert!(executed.receipt.programs.contains(&program.identity()));
    let elsewhere = ExecutionContext::for_tests(54);
    let again = compiled.execute(None, options(), &elsewhere).unwrap();
    assert!(again.receipt.programs.contains(&program.identity()));
    let recompiled = compile_preference_decision_across_studies(
        &[&benefit, &harm],
        &contract(1.2),
        distinct(2000, 7),
        &elsewhere,
    )
    .unwrap();
    assert_eq!(recompiled.joint_law().unwrap().fact().program(), program.identity());

    // Another seed or draw count is another program, and so another decision plan.
    for other in [distinct(2000, 8), distinct(2001, 7)] {
        let other = compile_preference_decision_across_studies(
            &[&benefit, &harm],
            &contract(1.2),
            other,
            &ctx,
        )
        .unwrap();
        assert_ne!(other.joint_law().unwrap().fact().program(), program.identity());
        assert_ne!(other.plan_views()[0].identity(), compiled.plan_views()[0].identity());
    }
}

#[test]
fn the_bound_table_draws_the_law_the_chain_compiled_after_fitting_draws() {
    let ctx = ExecutionContext::for_tests(61);
    let [benefit, harm] = separate_studies();
    let compiled = compile_preference_decision_across_studies(
        &[&benefit, &harm],
        &contract(1.2),
        distinct(2000, 7),
        &ctx,
    )
    .unwrap();
    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let Some(NativeDecisionSource::JointLaw(law)) = &executed.source else {
        panic!("one aligned law");
    };
    assert_eq!(law.as_ref(), &compiled_after_fitting(&compiled, &executed, &ctx));

    // Under shared evidence, with the declared cross block in the table.
    let table = shared_table();
    let shared_benefit = dose_study(&table, 3, 1);
    let shared_harm = dose_study(&table, 3, 2);
    let shared = SharedEvidence {
        draws: 3000,
        seed: 11,
        cross: vec![CrossStudyCovariance { first: 0, second: 1, covariance: vec![1e-4; 4] }],
    };
    let compiled = compile_preference_decision_with_shared_evidence(
        &[&shared_benefit, &shared_harm],
        &contract(1.2),
        &shared,
        &ctx,
    )
    .unwrap();
    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let Some(NativeDecisionSource::JointLaw(law)) = &executed.source else {
        panic!("one aligned law");
    };
    assert_eq!(law.as_ref(), &compiled_after_fitting(&compiled, &executed, &ctx));
}

#[test]
fn a_table_beyond_the_work_budget_is_refused_at_compile() {
    let ctx = ExecutionContext::for_tests(5);
    let [benefit, harm] = separate_studies();
    let refused = compile_preference_decision_across_studies(
        &[&benefit, &harm],
        &contract(1.2),
        distinct(1 << 27, 7),
        &ctx,
    )
    .unwrap_err();
    let refusal = refused.refusal().expect("a compile refusal");
    assert_eq!((refusal.code, refusal.stage), ("moment_matched_table_work_budget", "compile"));
    assert_eq!(refusal.detail, "propagation_chain.family_table_work_limit");
}

/// Two posterior slots `x` and `y` of one deferred table, each declaring `factors`, in `groups`.
fn deferred_chain(groups: &[Vec<usize>], factors: [&str; 2]) -> PropagationChainDraftWire {
    let estimates = [("x", factors[0]), ("y", factors[1])]
        .map(|(id, factor)| {
            FamilyEstimateDecl::new(id, LawMeaning::Posterior)
                .with_factor(SharedFactorKind::Row, factor)
        })
        .to_vec();
    let table = deferred_family_table("table", &estimates, groups, &[], 400, 7).expect("declared");
    assert!(table.sources.iter().all(|source| source.values.is_empty()));
    table.draft(
        ChainReading::Posterior,
        "sum",
        vec![ChainStageWire::new(
            "sum",
            StageKind::DerivedQuantity,
            &["x", "y"],
            ChainOpWire::binary("sum"),
        )],
    )
}

#[test]
fn a_bound_table_is_the_family_table_of_its_moments_and_replays_as_any_record() {
    let ctx = ExecutionContext::for_tests(9);
    let draft = deferred_chain(&[vec![0], vec![1]], ["rows-a", "rows-b"]);
    let plan = compile_deferred_propagation_chain(draft, &ctx).expect("compiles");
    assert!(plan.is_deferred());
    let binding = ChainTableBinding { means: vec![1.0, 2.0], within: vec![vec![0.25], vec![0.04]] };
    let chain = plan.execute_bound(&binding, &ctx).expect("binds");
    assert_eq!(chain.lowered_program_identity(), plan.identity());

    // The same table drawn from the estimates directly, sealed as an ordinary chain.
    let estimates = [("x", 1.0, 0.5, "rows-a"), ("y", 2.0, 0.2, "rows-b")]
        .map(|(id, mean, spread, factor)| {
            FamilyEstimate::new(id, mean, spread, LawMeaning::Posterior)
                .with_factor(SharedFactorKind::Row, factor)
        })
        .to_vec();
    let table = family_table(
        "table",
        &estimates,
        &FamilyDependence::Covariance(vec![0.25, 0.0, 0.0, 0.04]),
        None,
        400,
        7,
    )
    .expect("the family table");
    let direct = PropagationChain::seal(
        table.draft(
            ChainReading::Posterior,
            "sum",
            vec![ChainStageWire::new(
                "sum",
                StageKind::DerivedQuantity,
                &["x", "y"],
                ChainOpWire::binary("sum"),
            )],
        ),
        &ctx,
    )
    .expect("seals");
    assert_eq!(chain.record(), direct.record());
    let bytes = chain.export().expect("exports");
    let consumed =
        consume_propagation_chain(&bytes, Some(chain.identity()), &ctx).expect("replays");
    assert_eq!(consumed.identity, chain.identity());
}

#[test]
fn a_deferred_program_is_bound_or_refused() {
    let ctx = ExecutionContext::for_tests(9);
    let draft = deferred_chain(&[vec![0], vec![1]], ["rows-a", "rows-b"]);
    let plan = compile_deferred_propagation_chain(draft.clone(), &ctx).expect("compiles");

    let refusal = plan.execute(&ctx).unwrap_err();
    assert_eq!(refusal.detail(), DEFERRED_SOURCE_UNBOUND);
    // The draft's record alone is neither compiled as an ordinary chain nor sealed nor exported.
    let refusal = compile_propagation_chain(draft.record.clone(), &ctx).unwrap_err();
    assert_eq!(refusal.detail(), DEFERRED_SOURCE_UNBOUND);
    let refusal = PropagationChain::seal(draft.record.clone(), &ctx).unwrap_err();
    assert_eq!(refusal.detail(), DEFERRED_SOURCE_UNBOUND);
    let Err(PropagationChainError::Refused(refusal)) = draft.record.export() else {
        panic!("a deferred draft is not exported");
    };
    assert_eq!(refusal.detail(), DEFERRED_SOURCE_UNBOUND);

    // A binding of the wrong shape.
    for binding in [
        ChainTableBinding { means: vec![1.0], within: vec![vec![0.25], vec![0.04]] },
        ChainTableBinding { means: vec![1.0, 2.0], within: vec![vec![0.25]] },
        ChainTableBinding { means: vec![1.0, 2.0], within: vec![vec![0.25, 0.0], vec![0.04]] },
    ] {
        let refusal = plan.execute_bound(&binding, &ctx).unwrap_err();
        assert_eq!(refusal.detail(), "propagation_chain.family_shape");
    }

    // A within-group covariance no table can carry: a correlation of two.
    let draft = deferred_chain(&[vec![0, 1]], ["rows", "rows"]);
    let plan = compile_deferred_propagation_chain(draft, &ctx).expect("compiles");
    let binding =
        ChainTableBinding { means: vec![1.0, 2.0], within: vec![vec![1.0, 2.0, 2.0, 1.0]] };
    let refusal = plan.execute_bound(&binding, &ctx).unwrap_err();
    assert_eq!(refusal.detail(), "propagation_chain.family_table_refused");

    // Cancellation while binding is not a verdict on the chain.
    let cancelled = ExecutionContext::for_tests(9);
    cancelled.cancellation.cancel();
    let binding =
        ChainTableBinding { means: vec![1.0, 2.0], within: vec![vec![1.0, 0.5, 0.5, 1.0]] };
    let refusal = plan.execute_bound(&binding, &cancelled).unwrap_err();
    assert_eq!(refusal.code(), "cancelled_no_claim");
}

#[test]
fn a_deferred_table_refuses_what_its_declaration_cannot_carry() {
    let decl = |id: &str, factor: &str| {
        FamilyEstimateDecl::new(id, LawMeaning::Posterior)
            .with_factor(SharedFactorKind::Row, factor)
    };
    let estimates = vec![decl("x", "rows-a"), decl("y", "rows-b")];
    let groups = vec![vec![0], vec![1]];
    let detail =
        |result: Result<_, ChainRefusal>| result.map(|_| ()).unwrap_err().detail().to_owned();
    // Fewer than two rows, groups that do not partition the estimates.
    assert_eq!(
        detail(deferred_family_table("t", &estimates, &groups, &[], 1, 7)),
        "propagation_chain.family_shape"
    );
    assert_eq!(
        detail(deferred_family_table("t", &estimates, &[vec![0]], &[], 400, 7)),
        "propagation_chain.family_shape"
    );
    assert_eq!(
        detail(deferred_family_table("t", &estimates, &[vec![0, 1], vec![1]], &[], 400, 7)),
        "propagation_chain.family_shape"
    );
    // A predictive estimate has no moment-matched law.
    let predictive = vec![FamilyEstimateDecl::new("x", LawMeaning::Predictive), decl("y", "b")];
    assert_eq!(
        detail(deferred_family_table("t", &predictive, &groups, &[], 400, 7)),
        "propagation_chain.family_meaning"
    );
    // A declared block between estimates that declare no shared evidence, and shared evidence
    // with no declared block.
    let block = DeferredCrossBlock { first: 0, second: 1, covariance: vec![0.01] };
    assert_eq!(
        detail(deferred_family_table("t", &estimates, &groups, &[block], 400, 7)),
        "propagation_chain.family_dependence"
    );
    let shared = vec![decl("x", "rows"), decl("y", "rows")];
    assert_eq!(
        detail(deferred_family_table("t", &shared, &groups, &[], 400, 7)),
        "propagation_chain.family_dependence"
    );
    // More rows than the table's work budget admits.
    assert_eq!(
        detail(deferred_family_table("t", &estimates, &groups, &[], 1 << 28, 7)),
        "propagation_chain.family_table_work_limit"
    );
}

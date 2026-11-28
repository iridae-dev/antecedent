//! The program a compiled decision states is the program that runs.
//!
//! A decision across preferences is compiled into layer four of each study's program, and the
//! plan and the receipt state that decision program's identity. Execution enters exactly that
//! program: the prepared handle of every structure holds it, every result it produces reports
//! its identity, and the receipt lists the identity each executed result reported beside the
//! compiled one. The ordinary program of the study, without the decision, is another program
//! and never runs in its place, whether the study was compiled fresh or from a caller's
//! prepared handle, for one study, for several studies joined by a chain, and for the
//! structures of a graph posterior.
//!
//! The receipt also binds what the execution read besides the plans: the evaluation options
//! and the measure over the preference set. A compiled design's refusal carries a registered
//! reason code.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::joint_panel::JointPanelError;
use antecedent::analysis::preference_decision::{
    AcceptedPreferences, ActionInput, ActionKind, Attribute, CompiledPreferenceDecision,
    DecisionCriterion, OutcomeDependence, PreferenceAction, PreferenceContract,
    PreferenceContractParts, PreferenceEvaluationOptions, PreferenceExpr, PreferenceParameter,
    PreferencePolicy, PreferenceRouteError, PreferenceSet, RowProvenance, StructuralPolicy,
    ValueModel, compile_preference_decision_across_prepared_studies,
    compile_preference_decision_across_studies, shape,
};
use antecedent::estimate::joint_panel::PanelError;
use antecedent::{BayesianConfig, InferenceMode, PreparedStudy, RefuteSuite, Study};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    SemanticDigest, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::adjacency_mask::{AdjacencyMask, set_edge};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_prob::InferenceDiagnostics;

/// The receipt's executed entries when every plan ran and reported its identity.
fn all_ran(programs: &[SemanticDigest]) -> Vec<Option<SemanticDigest>> {
    programs.iter().copied().map(Some).collect()
}

fn options() -> PreferenceEvaluationOptions {
    PreferenceEvaluationOptions::default()
}

fn wobble(i: u32, rate: f64) -> f64 {
    (f64::from(i) * rate).sin()
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64))
}

fn quantity(variable: &str, units: &str, population: &str, dose: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: population.into(),
        regime_id: format!("do(t={dose})"),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn parts(actions: Vec<PreferenceAction>, population: &str) -> PreferenceContractParts {
    PreferenceContractParts {
        actions,
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: population.into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    }
}

/// `benefit - price * cost` over the mean of `y` at the two doses and a known cost.
fn priced() -> PreferenceContract {
    let model = ValueModel {
        attributes: vec![
            Attribute { id: "benefit".into(), units: "points".into() },
            Attribute { id: "cost".into(), units: "usd".into() },
        ],
        preferences: PreferenceSet::new(
            vec![PreferenceParameter {
                id: "price".into(),
                units: "points/usd".into(),
                lower: 0.1,
                upper: 0.5,
                assumption_id: "assumption.price_of_benefit".into(),
            }],
            vec![],
        ),
        value: PreferenceExpr::difference(
            PreferenceExpr::Attribute(0),
            PreferenceExpr::product(PreferenceExpr::Parameter(0), PreferenceExpr::Attribute(1)),
        ),
        utility_units: "points".into(),
    };
    let action = |id: &str, dose: &str, cost: f64| PreferenceAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![
            ActionInput::Quantity(quantity("y", "points", "target", dose)),
            ActionInput::Fixed { value: cost, units: "usd".into() },
        ],
    };
    AcceptedPreferences::declared(model)
        .unwrap()
        .contract(parts(vec![action("low", "0.5", 0.5), action("high", "1.5", 1.5)], "target"))
}

/// `benefit - harm_rate * harm - excess_weight * max(harm - 1.2, 0)` over two outcomes of
/// separately estimated studies: a value not affine in the outcomes, so one aligned law.
fn kinked() -> PreferenceContract {
    let model = ValueModel {
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
            PreferenceExpr::product(PreferenceExpr::Parameter(1), shape::excess(1, 1.2)),
        ),
        utility_units: "points".into(),
    };
    let action = |id: &str, dose: &str| PreferenceAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![
            ActionInput::Quantity(quantity("b", "points", "patients", dose)),
            ActionInput::Quantity(quantity("h", "events", "patients", dose)),
        ],
    };
    AcceptedPreferences::declared(model)
        .unwrap()
        .contract(parts(vec![action("high", "1.5"), action("low", "0.5")], "patients"))
}

fn dose_table(outcome: &str, (intercept, slope): (f64, f64), rate: f64) -> TabularData {
    let rows = 200_u32;
    let t: Vec<f64> = (0..rows).map(|i| 2.0 * f64::from(i) / f64::from(rows - 1)).collect();
    let y: Vec<f64> = (0..rows)
        .zip(&t)
        .map(|(i, dose)| intercept + slope * dose + 0.6 * wobble(i, rate))
        .collect();
    TabularData::from_f64_columns([("t", t.as_slice()), (outcome, y.as_slice())]).unwrap()
}

/// A dose-response study of column 1 of a two-column table.
fn dose_study(data: &TabularData, inference: InferenceMode) -> Study {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.5, 1.5])),
            ),
        })))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        // A study of a whole table declares that table as its rows' source.
        .row_provenance(RowProvenance::of_table(data).unwrap())
        .build()
        .unwrap()
}

/// The identity of the ordinary program of `study`, without a decision: what a plain
/// prepared estimate reports.
fn ordinary_identity(study: &Study, ctx: &ExecutionContext) -> Option<SemanticDigest> {
    study.prepare(ctx).unwrap().estimate_retained(ctx).unwrap().lowered_program_identity()
}

/// Every structure that compiled: its stated plan identity, and the identity its prepared
/// handle's result reports.
fn stated_and_ran(
    compiled: &CompiledPreferenceDecision,
    ctx: &ExecutionContext,
) -> Vec<(SemanticDigest, Option<SemanticDigest>)> {
    compiled
        .structures()
        .iter()
        .filter_map(|structure| {
            let view = structure.plan_view()?;
            let handle: &PreparedStudy = structure.prepared().expect("a compiled structure runs");
            let ran = handle.estimate_retained(ctx).unwrap().lowered_program_identity();
            Some((view.identity(), ran))
        })
        .collect()
}

#[test]
fn one_study_executes_the_decision_program_its_receipt_names() {
    let study = dose_study(&dose_table("y", (0.5, 1.5), 1.7), InferenceMode::Frequentist);
    let ctx = ExecutionContext::for_tests(3);
    let ordinary = ordinary_identity(&study, &ctx);
    assert!(ordinary.is_some());

    let compiled = study.compile_for_preference_decision(&priced(), &ctx).unwrap();
    let pairs = stated_and_ran(&compiled, &ctx);
    let [(stated, ran)] = pairs.as_slice() else {
        panic!("one structure");
    };
    let (stated, ran) = (*stated, *ran);
    assert_eq!(ran, Some(stated), "the handle runs the stated decision program");
    assert_ne!(ordinary, Some(stated), "the decision is in the stated program's layer four");

    let executed = compiled.execute(None, options(), &ctx).unwrap();
    assert_eq!(executed.receipt.programs, [stated]);
    assert_eq!(executed.receipt.executed, [Some(stated)], "the executed result reports it");

    // A caller's prepared handle executes the decision program too; the caller's own handle
    // keeps its ordinary program.
    let prepared = study.prepare(&ctx).unwrap();
    let held = prepared.compile_for_preference_decision(&priced(), &ctx).unwrap();
    let pairs = stated_and_ran(&held, &ctx);
    let [(stated, ran)] = pairs.as_slice() else {
        panic!("one structure");
    };
    let (stated, ran) = (*stated, *ran);
    assert_eq!(ran, Some(stated));
    assert_ne!(ordinary, Some(stated));
    let executed = held.execute(None, options(), &ctx).unwrap();
    assert_eq!(executed.receipt.executed, all_ran(&executed.receipt.programs));
    assert_eq!(prepared.estimate_retained(&ctx).unwrap().lowered_program_identity(), ordinary);
}

#[test]
fn studies_joined_by_a_chain_execute_their_decision_programs_and_the_chain() {
    let ctx = ExecutionContext::for_tests(5);
    let benefit = dose_study(&dose_table("b", (2.0, 1.2), 1.7), bayes());
    let harm = dose_study(&dose_table("h", (0.5, 0.5), 2.9), bayes());
    let dependence = OutcomeDependence::DistinctEvidence { draws: 2000, seed: 7 };
    let compiled =
        compile_preference_decision_across_studies(&[&benefit, &harm], &kinked(), dependence, &ctx)
            .unwrap();
    let pairs = stated_and_ran(&compiled, &ctx);
    assert_eq!(pairs.len(), 2);
    for ((stated, ran), study) in pairs.iter().zip([&benefit, &harm]) {
        assert_eq!(*ran, Some(*stated));
        assert_ne!(ordinary_identity(study, &ctx), Some(*stated));
    }
    let chain = compiled.joint_law().expect("a declared chain").fact().program();
    let executed = compiled.execute(None, options(), &ctx).unwrap();
    assert_eq!(executed.receipt.programs, [pairs[0].0, pairs[1].0, chain]);
    assert_eq!(executed.receipt.executed, all_ran(&executed.receipt.programs));

    // The same through the callers' prepared handles.
    let handles = [benefit.prepare(&ctx).unwrap(), harm.prepare(&ctx).unwrap()];
    let held = compile_preference_decision_across_prepared_studies(
        &[&handles[0], &handles[1]],
        &kinked(),
        (dependence, &[]),
        &ctx,
    )
    .unwrap();
    for (stated, ran) in stated_and_ran(&held, &ctx) {
        assert_eq!(ran, Some(stated));
    }
    let executed = held.execute(None, options(), &ctx).unwrap();
    assert_eq!(executed.receipt.executed, all_ran(&executed.receipt.programs));
}

/// A confounded table over `t, y, z`.
fn confounded_table() -> TabularData {
    let n = 300_u32;
    let z: Vec<f64> = (0..n).map(|i| wobble(i, 0.91)).collect();
    let t: Vec<f64> = (0..n).zip(&z).map(|(i, z)| 1.0 + 0.4 * z + 0.5 * wobble(i, 2.3)).collect();
    let y: Vec<f64> = (0..n)
        .zip(t.iter().zip(&z))
        .map(|(i, (t, z))| 0.5 + 1.5 * t + z + 0.3 * wobble(i, 1.3))
        .collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

fn mask(edges: &[(usize, usize)]) -> AdjacencyMask {
    edges.iter().fold(AdjacencyMask::EMPTY, |mask, &(from, to)| set_edge(mask, 3, from, to, true))
}

/// DAG atoms `z -> t -> y <- z` (0.6) and `t -> y` (0.4).
fn dag_posterior() -> GraphPosterior {
    GraphPosterior::from_atoms(
        3,
        vec![0.6, 0.4],
        vec![mask(&[(2, 0), (0, 1), (2, 1)]), mask(&[(0, 1)])],
        vec![0.0; 9],
        vec![0.0; 9],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("preference_program_identity"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Dag)
}

#[test]
fn the_structures_of_a_graph_posterior_execute_their_decision_programs() {
    antecedent::support::caller_thread()
        .stack_size(64 * 1024 * 1024)
        .spawn(posterior_body)
        .expect("spawn the structural body")
        .join()
        .expect("the structural body completed");
}

fn posterior_body() {
    let ctx = ExecutionContext::for_tests(7);
    let study = Study::tabular(confounded_table())
        .graph_posterior(dag_posterior())
        .query(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.5, 1.5])),
            ),
        })))
        .inference(bayes())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let compiled = study.compile_for_preference_decision(&priced(), &ctx).unwrap();
    let pairs = stated_and_ran(&compiled, &ctx);
    assert_eq!(pairs.len(), 2, "both atoms identify the curve");
    for (stated, ran) in &pairs {
        assert_eq!(*ran, Some(*stated));
    }
    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let stated: Vec<SemanticDigest> = pairs.iter().map(|(stated, _)| *stated).collect();
    assert_eq!(executed.receipt.programs, stated);
    assert_eq!(executed.receipt.executed, all_ran(&stated));
}

#[test]
fn the_receipt_binds_the_options_and_measure_the_execution_read() {
    let study = dose_study(&dose_table("y", (0.5, 1.5), 1.7), InferenceMode::Frequentist);
    let ctx = ExecutionContext::for_tests(9);
    let compiled = study.compile_for_preference_decision(&priced(), &ctx).unwrap();
    let first = compiled.execute(None, options(), &ctx).unwrap().receipt;
    let again = compiled.execute(None, options(), &ctx).unwrap().receipt;
    assert_eq!(first, again, "the same execution gives the same receipt");
    assert_eq!(first.measure, None);
    let mut looser = options();
    looser.exact.tie_tolerance = 1e-3;
    looser.certified.max_boxes = 64;
    let other = compiled.execute(None, looser, &ctx).unwrap().receipt;
    assert_ne!(other.options, first.options, "other tolerances are another execution");
    assert_eq!(other.programs, first.programs);
}

#[test]
fn a_design_refusal_carries_a_registered_code() {
    let code = |error: JointPanelError| PreferenceRouteError::from(error).code().map(str::to_owned);
    assert_eq!(
        code(JointPanelError::Unsupported("no design proof".into())).as_deref(),
        Some("route_not_supported")
    );
    assert_eq!(
        code(JointPanelError::Engine(PanelError::Cancelled)).as_deref(),
        Some("cancelled_no_claim")
    );
    assert_eq!(
        code(JointPanelError::Engine(PanelError::Resource("bytes".into()))).as_deref(),
        Some("execution_resource_limit")
    );
    for error in [
        JointPanelError::Compile("roles disagree".into()),
        JointPanelError::Engine(PanelError::Invalid("shape".into())),
        JointPanelError::Engine(PanelError::Numerical("curvature".into())),
    ] {
        assert_eq!(code(error).as_deref(), Some("invalid_argument"));
    }
}

//! A decision across preferences compiled over graph posteriors whose atoms are not fully
//! oriented DAGs.
//!
//! Fixture. A confounded table `y = 0.5 + 1.5 t + z + noise`, `t = 1 + 0.4 z + noise`, and a
//! mean response curve of `y` under `do(t = 0.5)` and `do(t = 1.5)`. The mixed-graph route
//! evaluates its identifying functional with the discrete plug-in, so the ADMG atoms read a
//! coded version of the table: a binary `z`, `t` at the two doses (more often high when `z` is
//! one) and the noise on three levels.
//!
//! * ADMG atoms with weights `(0.5, 0.3, 0.2)`: `t -> y <- z`; `t -> y` with a latent common
//!   cause `z <-> y`; and a directed cycle, which is no acyclic mixed graph, so its mass is
//!   unidentified. Each acyclic atom compiles as an ADMG study of its own structure and
//!   answers as that mixed graph compiled on its own; contemporaneous temporal ADMG atoms,
//!   whose identification is the static ADMG functional, give the same answer. On the
//!   continuous table the mixed-graph route cannot evaluate the curve, and the compile is
//!   refused naming the atom kind.
//! * A CPDAG atom `z - t -> y`: its completions `z -> t -> y` and `t -> z, t -> y` compile one
//!   by one as DAG studies. The class weight splits evenly between them as an enumeration
//!   weight, not a probability, so `BayesOverStructures` is refused at compile while the
//!   invariance policy reads them as graph-dependent claims. Each completion answers as its
//!   DAG compiled on its own.
//!
//! Every compiled decision agrees with the result-level functions on the responses the
//! execution retained.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::native_claims::NativeResponseClaim;
use antecedent::analysis::preference_decision::{
    self as pd, AcceptedPreferences, ActionInput, ActionKind, Attribute,
    CompiledPreferenceDecision, DecisionCriterion, PreferenceAction, PreferenceAtomStatus,
    PreferenceContract, PreferenceContractParts, PreferenceEvaluationOptions, PreferenceExpr,
    PreferenceParameter, PreferencePolicy, PreferenceSet, StructuralPolicy,
    StructuralPreferenceEvaluation, StructureIdentification, ValueModel,
};
use antecedent::planner::lowered::CarrierDemand;
use antecedent::result::StructuralWeightBasis;
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_discovery::adjacency_mask::{AdjacencyMask, admg_from_adjacency_mask, set_edge};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_prob::InferenceDiagnostics;

const T: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "actual={actual}, expected={expected}"
    );
}

fn options() -> PreferenceEvaluationOptions {
    PreferenceEvaluationOptions::default()
}

fn wobble(i: u32, rate: f64) -> f64 {
    (f64::from(i) * rate).sin()
}

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

/// The coded table the mixed-graph route evaluates with its discrete plug-in.
fn coded_table() -> TabularData {
    let n = 300_u32;
    let z: Vec<f64> = (0..n).map(|i| f64::from(i % 2)).collect();
    let t: Vec<f64> = (0..n)
        .zip(&z)
        .map(|(i, z)| if f64::from((i / 2) % 5) < 2.0 + 2.0 * z { 1.5 } else { 0.5 })
        .collect();
    let y: Vec<f64> = (0..n)
        .zip(t.iter().zip(&z))
        .map(|(i, (t, z))| 0.5 + 1.5 * t + z + 0.3 * (f64::from((i / 10) % 3) - 1.0))
        .collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

/// Both directed bits of a pair encode a bidirected edge (ADMG) or an undirected one (CPDAG).
fn mask(edges: &[(u32, u32)]) -> AdjacencyMask {
    edges.iter().fold(AdjacencyMask::EMPTY, |mask, &(from, to)| {
        set_edge(mask, 3, usize::try_from(from).unwrap(), usize::try_from(to).unwrap(), true)
    })
}

fn posterior_of(
    structures: &[&[(u32, u32)]],
    weights: &[f64],
    kind: GraphPosteriorAtomKind,
) -> GraphPosterior {
    GraphPosterior::from_atoms(
        3,
        weights.to_vec(),
        structures.iter().map(|edges| mask(edges)).collect::<Vec<_>>(),
        vec![0.0; 9],
        vec![0.0; 9],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("preference_atom_kinds"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(kind)
}

fn study(data: &TabularData, posterior: GraphPosterior) -> Study {
    Study::tabular(data.clone())
        .graph_posterior(posterior)
        .query(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(Y),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(T),
                GridSpec::Values(Arc::from([0.5, 1.5])),
            ),
        })))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

fn quantity(dose: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
        role: QuantityRole::Outcome,
        units: "points".into(),
        population_id: "target".into(),
        regime_id: format!("do(t={dose})"),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

/// `benefit - price * cost`, the costs of `low` and `high` differing by one.
fn contract(lower: f64, upper: f64, structural_policy: StructuralPolicy) -> PreferenceContract {
    let model = ValueModel {
        attributes: vec![
            Attribute { id: "benefit".into(), units: "points".into() },
            Attribute { id: "cost".into(), units: "usd".into() },
        ],
        preferences: PreferenceSet::new(
            vec![PreferenceParameter {
                id: "price".into(),
                units: "points/usd".into(),
                lower,
                upper,
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
            ActionInput::Quantity(quantity(dose)),
            ActionInput::Fixed { value: cost, units: "usd".into() },
        ],
    };
    AcceptedPreferences::declared(model).unwrap().contract(PreferenceContractParts {
        actions: vec![action("low", "0.5", 0.5), action("high", "1.5", 1.5)],
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy,
        preference_policy: PreferencePolicy::RequireInvariantBestAction,
    })
}

/// Each identified structure's own answer: the decision on the claim of the response the
/// execution retained for it.
fn atom_answers(
    compiled: &CompiledPreferenceDecision,
    executed: &pd::PreferenceDecisionExecution,
    contract: &PreferenceContract,
    data: &TabularData,
    ctx: &ExecutionContext,
) -> Vec<pd::PreferenceEvaluation> {
    let context = compiled.response_context(0, ctx).unwrap();
    executed
        .mixture
        .as_ref()
        .unwrap()
        .atoms
        .iter()
        .filter_map(|atom| atom.response.as_ref())
        .map(|response| {
            let claim =
                NativeResponseClaim::from_response(response, data.schema(), &context).unwrap();
            pd::decide_claim(contract, &claim, None, options()).unwrap().evaluation
        })
        .collect()
}

/// The answer of `graph` compiled on its own, as a study of that single structure.
fn single_graph_answer(
    data: &TabularData,
    graph: impl antecedent::IntoGraphInput,
    contract: &PreferenceContract,
    ctx: &ExecutionContext,
) -> pd::PreferenceEvaluation {
    let single = Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(Y),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(T),
                GridSpec::Values(Arc::from([0.5, 1.5])),
            ),
        })))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let compiled = single.compile_for_preference_decision(contract, ctx).unwrap();
    let executed = compiled.execute(None, options(), ctx).unwrap();
    executed.evaluation.single().cloned().expect("one structure answers once")
}

/// Graph-posterior compiles and estimations recurse deeply; run the body on a large stack.
fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("compiled-preference-atoms".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("spawn the structural body")
        .join()
        .expect("the structural body completed");
}

#[test]
fn admg_atoms_compile_as_mixed_graph_studies_and_keep_their_unidentified_mass() {
    on_large_stack(admg_body);
}

fn admg_body() {
    let covariate: &[(u32, u32)] = &[(T, Y), (Z, Y)];
    let latent: &[(u32, u32)] = &[(T, Y), (Z, Y), (Y, Z)];
    let cycle: &[(u32, u32)] = &[(T, Y), (Y, Z), (Z, T)];
    let posterior =
        posterior_of(&[covariate, latent, cycle], &[0.5, 0.3, 0.2], GraphPosteriorAtomKind::Admg);
    let keys = posterior.graph_keys().clone();
    let ctx = ExecutionContext::for_tests(31);
    let invariant = contract(0.1, 0.5, StructuralPolicy::RequireInvariantBestAction);

    // On the continuous table the mixed-graph route cannot evaluate the curve: refused at
    // compile, naming the atom kind.
    let refused = study(&confounded_table(), posterior.clone())
        .compile_for_preference_decision(&invariant, &ctx)
        .unwrap_err();
    let refusal = refused.refusal().expect("a compile refusal");
    assert_eq!((refusal.code, refusal.stage), ("route_not_supported", "compile"));
    assert_eq!(
        refusal.detail,
        antecedent::reason::reason_details::preference_compile::ATOM_KIND_NOT_COMPILED
    );

    let data = coded_table();
    let structured = study(&data, posterior);
    let compiled = structured.compile_for_preference_decision(&invariant, &ctx).unwrap();

    let structures = compiled.structures();
    assert_eq!(structures.len(), 3);
    for (structure, (key, weight)) in structures.iter().zip(keys.iter().zip([0.5, 0.3, 0.2])) {
        assert_eq!(structure.key(), Some(*key));
        assert_eq!(structure.class_key(), None);
        close(structure.weight(), weight, 1e-12);
    }
    assert!(structures[..2].iter().all(|s| s.plan_view().is_some()));
    assert!(matches!(structures[2].identification(), StructureIdentification::NotIdentified(_)));
    close(compiled.unidentified_mass(), 0.2, 1e-12);
    for view in compiled.plan_views() {
        assert_eq!(
            view.evidence().decision().unwrap().carrier(),
            CarrierDemand::ParameterPosterior
        );
    }

    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let mixture = executed.mixture.as_ref().expect("a structural mixture");
    assert_eq!(mixture.weight_basis, StructuralWeightBasis::PosteriorProbability);
    close(mixture.identified_mass + mixture.unevaluable_mass, 0.8, 1e-12);
    close(mixture.unidentified_mass, 0.2, 1e-12);
    let Some(StructuralPreferenceEvaluation::Exact(result)) = executed.evaluation.structures()
    else {
        panic!("an affine expectation is answered exactly");
    };
    close(result.unidentified_mass.unwrap(), 0.2, 1e-9);
    let unidentified = format!("{}{}", pd::GRAPH_ATOM_PREFIX, keys[2]);
    assert!(
        result.atoms.iter().any(|atom| atom.id == unidentified
            && matches!(atom.status, PreferenceAtomStatus::Unidentified))
    );

    // The result-level functions on the retained structures give the same evaluation.
    let claims = pd::structure_claims(
        mixture,
        &[],
        data.schema(),
        &compiled.response_context(0, &ctx).unwrap(),
        StructuralPolicy::RequireInvariantBestAction,
    )
    .unwrap();
    let by_hand = pd::decide_across_structures(&invariant, &claims, None, options()).unwrap();
    assert_eq!(executed.evaluation.structures(), Some(&by_hand));

    // Each acyclic atom answers as its mixed graph compiled on its own.
    let answers = atom_answers(&compiled, &executed, &invariant, &data, &ctx);
    assert_eq!(answers.len(), 2);
    for (edges, answer) in [covariate, latent].iter().zip(&answers) {
        let admg = admg_from_adjacency_mask(mask(edges), 3).unwrap();
        assert_eq!(&single_graph_answer(&data, admg, &invariant, &ctx), answer);
    }

    // Contemporaneous temporal ADMG atoms are the static ADMG functional: the same answer.
    let temporal = posterior_of(
        &[covariate, latent, cycle],
        &[0.5, 0.3, 0.2],
        GraphPosteriorAtomKind::TemporalAdmg,
    );
    let temporal =
        study(&data, temporal).compile_for_preference_decision(&invariant, &ctx).unwrap();
    close(temporal.unidentified_mass(), 0.2, 1e-12);
    let rerun = temporal.execute(None, options(), &ctx).unwrap();
    assert_eq!(rerun.evaluation.structures(), executed.evaluation.structures());
    assert_eq!(atom_answers(&temporal, &rerun, &invariant, &data, &ctx), answers);

    // Bayes over structures reads the genuine posterior weights of mixed-graph atoms.
    let bayes = contract(0.1, 0.5, StructuralPolicy::BayesOverStructures);
    assert!(structured.compile_for_preference_decision(&bayes, &ctx).is_ok());
}

#[test]
fn a_class_atom_compiles_completion_by_completion_without_making_counts_probabilities() {
    on_large_stack(class_body);
}

fn class_body() {
    let data = confounded_table();
    // `z - t` undirected (both bits), `t -> y` directed.
    let class: &[(u32, u32)] = &[(Z, T), (T, Z), (T, Y)];
    let posterior = posterior_of(&[class], &[1.0], GraphPosteriorAtomKind::Cpdag);
    let key = posterior.graph_keys()[0];
    let ctx = ExecutionContext::for_tests(37);
    let structured = study(&data, posterior);

    // Completion weights are not probabilities: Bayes over structures is refused at compile.
    let refused = structured
        .compile_for_preference_decision(
            &contract(0.1, 0.5, StructuralPolicy::BayesOverStructures),
            &ctx,
        )
        .unwrap_err();
    let refusal = refused.refusal().expect("a compile refusal");
    assert_eq!(refusal.stage, "compile");
    assert_eq!(
        refusal.detail,
        antecedent::reason::reason_details::decision_adapters::COMPLETION_COUNT_NOT_PROBABILITY
    );

    let invariant = contract(0.1, 0.5, StructuralPolicy::RequireInvariantBestAction);
    let compiled = structured.compile_for_preference_decision(&invariant, &ctx).unwrap();
    let structures = compiled.structures();
    assert_eq!(structures.len(), 2, "the two DAG members of the class");
    for structure in structures {
        assert_eq!(structure.class_key(), Some(key));
        close(structure.weight(), 0.5, 1e-12);
        assert!(structure.plan_view().is_some(), "t -> y is identified in both members");
    }
    assert_ne!(structures[0].key(), structures[1].key());
    close(compiled.unidentified_mass(), 0.0, 1e-12);

    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let mixture = executed.mixture.as_ref().expect("a structural mixture");
    assert_eq!(mixture.weight_basis, StructuralWeightBasis::CompletionEnumeration);
    assert_eq!(mixture.atoms.iter().filter(|atom| atom.response.is_some()).count(), 2);
    let Some(StructuralPreferenceEvaluation::Exact(result)) = executed.evaluation.structures()
    else {
        panic!("an affine expectation is answered exactly");
    };
    assert!(result.atoms.iter().all(|atom| atom.probability.is_none()), "no probabilities");
    let claims = pd::structure_claims(
        mixture,
        &[],
        data.schema(),
        &compiled.response_context(0, &ctx).unwrap(),
        StructuralPolicy::RequireInvariantBestAction,
    )
    .unwrap();
    let by_hand = pd::decide_across_structures(&invariant, &claims, None, options()).unwrap();
    assert_eq!(executed.evaluation.structures(), Some(&by_hand));

    // Each completion answers as that DAG compiled on its own.
    let dag = |edges: &[(u32, u32)]| {
        let mut graph = Dag::with_variables(3);
        for &(from, to) in edges {
            graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        graph
    };
    let mut singles: Vec<pd::PreferenceEvaluation> = [&[(Z, T), (T, Y)][..], &[(T, Z), (T, Y)]]
        .iter()
        .map(|edges| single_graph_answer(&data, dag(edges), &invariant, &ctx))
        .collect();
    for answer in atom_answers(&compiled, &executed, &invariant, &data, &ctx) {
        let found = singles.iter().position(|single| *single == answer);
        singles.remove(found.expect("every completion answers as its DAG alone"));
    }
    assert!(singles.is_empty());
}

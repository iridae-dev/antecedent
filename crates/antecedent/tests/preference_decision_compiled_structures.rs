//! A decision across preferences compiled over the structures of a graph posterior.
//!
//! Fixture. A confounded table `y = 0.5 + 1.5 t + z + noise`, `t = 1 + 0.4 z + noise`, and a
//! posterior over three structures with weights `(0.5, 0.3, 0.2)`: `t -> y <- z`,
//! `z -> t -> y <- z`, and a cyclic `t <-> y` pair under which the effect is not identified.
//! The compiler compiles and prepares each distinct structure as a study of its own structure:
//! the unidentified mass is known before anything is fitted, each identified structure's own
//! response is retained by the execution, and Bayes over structures mixes the two identified
//! structures with weights 0.5 and 0.3 without renormalizing.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::native_claims::{NativeDecisionSource, NativeResponseClaim};
use antecedent::analysis::preference_decision::{
    self as pd, AcceptedPreferences, ActionInput, ActionKind, Attribute,
    CompiledPreferenceDecision, DecisionCriterion, PreferenceAction, PreferenceAtomStatus,
    PreferenceContract, PreferenceContractParts, PreferenceEvaluationOptions, PreferenceExpr,
    PreferenceParameter, PreferencePolicy, PreferenceSet, PreferenceVerdict, StructuralPolicy,
    StructuralPreferenceEvaluation, StructuralPreferenceVerdict, StructureIdentification,
    ValueModel,
};
use antecedent::design::decision_contract::{SourceRepresentation, SourceRequirement};
use antecedent::planner::lowered::CarrierDemand;
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_discovery::adjacency_mask::{AdjacencyMask, set_edge};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_io::distribution_artifact::DistributionArtifact;
use antecedent_prob::InferenceDiagnostics;

const T: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;
const COVARIATE: [(u32, u32); 2] = [(T, Y), (Z, Y)];
const CONFOUNDER: [(u32, u32); 3] = [(Z, T), (Z, Y), (T, Y)];
const CYCLIC: [(u32, u32); 2] = [(T, Y), (Y, T)];

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

fn mask(edges: &[(u32, u32)]) -> AdjacencyMask {
    edges.iter().fold(AdjacencyMask::EMPTY, |mask, &(from, to)| {
        set_edge(mask, 3, usize::try_from(from).unwrap(), usize::try_from(to).unwrap(), true)
    })
}

fn posterior_of(structures: &[&[(u32, u32)]], weights: &[f64]) -> GraphPosterior {
    GraphPosterior::from_atoms(
        3,
        weights.to_vec(),
        structures.iter().map(|edges| mask(edges)).collect::<Vec<_>>(),
        vec![0.0; 9],
        vec![0.0; 9],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("preference_structures"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Dag)
}

fn study(data: &TabularData, posterior: GraphPosterior, inference: InferenceMode) -> Study {
    Study::tabular(data.clone())
        .graph_posterior(posterior)
        .query(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(Y),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(T),
                GridSpec::Values(Arc::from([0.5, 1.5])),
            ),
        })))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64))
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
        preferences: PreferenceSet {
            parameters: vec![PreferenceParameter {
                id: "price".into(),
                units: "points/usd".into(),
                lower,
                upper,
                assumption_id: "assumption.price_of_benefit".into(),
            }],
            relations: vec![],
        },
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

/// Graph-posterior compiles and estimations recurse deeply; run the body on a large stack.
fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("compiled-preference-structures".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("spawn the structural body")
        .join()
        .expect("the structural body completed");
}

/// Mean of column `column` of an equally weighted law.
fn draw_mean(law: &DistributionArtifact, column: usize) -> f64 {
    let [n, width] = law.shape();
    let total: f64 = (0..n).map(|draw| law.draws()[draw * width + column]).sum();
    total / f64::from(u32::try_from(n).unwrap())
}

/// The gain `m(1.5) - m(0.5)` of every evaluated structure, read from the responses the
/// execution retained, in structure order.
fn structure_gains(
    compiled: &CompiledPreferenceDecision,
    data: &TabularData,
    ctx: &ExecutionContext,
) -> Vec<(f64, f64, f64)> {
    let executed = compiled.execute(None, options(), ctx).unwrap();
    let mixture = executed.mixture.expect("a structural mixture");
    let context = compiled.response_context(0, ctx).unwrap();
    let joint = SourceRequirement {
        any_of: vec![SourceRepresentation::JointDraws],
        sampled_needs_error_receipt: false,
    };
    mixture
        .atoms
        .iter()
        .filter_map(|atom| atom.response.as_ref().map(|response| (atom.weight, response)))
        .map(|(weight, response)| {
            let claim =
                NativeResponseClaim::from_response(response, data.schema(), &context).unwrap();
            let NativeDecisionSource::JointLaw(law) = claim.decision_source(&joint).unwrap().source
            else {
                panic!("a Bayesian structure retains aligned draws");
            };
            (weight, draw_mean(&law, 0), draw_mean(&law, 1))
        })
        .collect()
}

#[test]
fn structures_are_compiled_and_their_unidentified_mass_is_known_before_fitting() {
    on_large_stack(compile_body);
}

fn compile_body() {
    let data = confounded_table();
    let posterior = posterior_of(&[&COVARIATE[..], &CONFOUNDER[..], &CYCLIC[..]], &[0.5, 0.3, 0.2]);
    let keys = posterior.graph_keys().clone();
    let structured = study(&data, posterior, bayes());
    let ctx = ExecutionContext::for_tests(31);
    let compiled = structured
        .compile_for_preference_decision(
            &contract(0.1, 0.5, StructuralPolicy::BayesOverStructures),
            &ctx,
        )
        .unwrap();
    assert_eq!(compiled.demand().representation, SourceRepresentation::JointDraws);
    assert_eq!(compiled.demand().carrier, Some(CarrierDemand::ParameterPosterior));
    let structures = compiled.structures();
    assert_eq!(structures.len(), 3);
    for (structure, (key, weight)) in structures.iter().zip(keys.iter().zip([0.5, 0.3, 0.2])) {
        assert_eq!(structure.key(), Some(*key));
        close(structure.weight(), weight, 1e-12);
    }
    assert!(structures[..2].iter().all(|s| s.plan_view().is_some()));
    assert!(matches!(structures[2].identification(), StructureIdentification::NotIdentified(_)));
    assert!(structures[2].plan_view().is_none());
    close(compiled.unidentified_mass(), 0.2, 1e-12);
    for view in compiled.plan_views() {
        let decision = view.evidence().decision().unwrap();
        assert_eq!(decision.carrier(), CarrierDemand::ParameterPosterior);
        assert_eq!(decision.contract_identity(), Some(compiled.bound_identity()));
    }

    // A frequentist posterior cannot supply each structure's joint law: refused at compile.
    let frequentist = study_of_frequentist(&data);
    let refused = frequentist
        .compile_for_preference_decision(
            &contract(0.1, 0.5, StructuralPolicy::BayesOverStructures),
            &ctx,
        )
        .unwrap_err();
    assert!(refused.compiler().is_some(), "{refused}");

    // No structure identifies the curve: refused at compile with the whole mass.
    let cyclic = study(&data, posterior_of(&[&CYCLIC[..]], &[1.0]), bayes());
    let refused = cyclic
        .compile_for_preference_decision(
            &contract(0.1, 0.5, StructuralPolicy::BayesOverStructures),
            &ctx,
        )
        .unwrap_err();
    let refusal = refused.refusal().expect("an identification refusal");
    assert_eq!(refusal.code, "effect_not_identified");
    assert_eq!(refusal.stage, "compile");
    assert!(refusal.supplied.as_deref().is_some_and(|s| s.starts_with("unidentified mass 1")));
}

fn study_of_frequentist(data: &TabularData) -> Study {
    let posterior = posterior_of(&[&COVARIATE[..], &CONFOUNDER[..], &CYCLIC[..]], &[0.5, 0.3, 0.2]);
    study(data, posterior, InferenceMode::Frequentist)
}

#[test]
fn bayes_over_compiled_structures_keeps_the_unidentified_mass() {
    on_large_stack(decide_body);
}

#[allow(
    clippy::too_many_lines,
    reason = "one structural story: the compiled structures, both policies, the agreements"
)]
fn decide_body() {
    let data = confounded_table();
    let posterior = posterior_of(&[&COVARIATE[..], &CONFOUNDER[..], &CYCLIC[..]], &[0.5, 0.3, 0.2]);
    let keys = posterior.graph_keys().clone();
    let study = study(&data, posterior, bayes());
    let ctx = ExecutionContext::for_tests(31);
    let probe = study
        .compile_for_preference_decision(
            &contract(0.1, 0.5, StructuralPolicy::BayesOverStructures),
            &ctx,
        )
        .unwrap();
    let gains = structure_gains(&probe, &data, &ctx);
    assert_eq!(gains.len(), 2);
    // Bayes over structures, not renormalized: the combined gain is `0.5 g1 + 0.3 g2` and the
    // combined price weight `0.8`, so `high` leads below `(0.5 g1 + 0.3 g2) / 0.8`.
    let mixed_gain: f64 = gains.iter().map(|(w, low, high)| w * (high - low)).sum();
    let switch = mixed_gain / 0.8;

    let bayes_contract = contract(0.1, 0.5 * switch, StructuralPolicy::BayesOverStructures);
    let compiled = study.compile_for_preference_decision(&bayes_contract, &ctx).unwrap();
    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let Some(StructuralPreferenceEvaluation::Exact(result)) = executed.evaluation.structures()
    else {
        panic!("an affine expectation is answered exactly");
    };
    close(result.unidentified_mass.unwrap(), 0.2, 1e-9);
    close(result.evaluated_mass.unwrap(), 0.8, 1e-9);
    let unidentified = format!("{}{}", pd::GRAPH_ATOM_PREFIX, keys[2]);
    assert!(
        result.atoms.iter().any(|atom| atom.id == unidentified
            && matches!(atom.status, PreferenceAtomStatus::Unidentified))
    );
    assert!(matches!(result.verdict, StructuralPreferenceVerdict::InsufficientScience(_)));
    let combined = result.combined.as_ref().unwrap();
    assert_eq!(combined.verdict, PreferenceVerdict::NecessarilyBest("high".into()));
    let high = combined.actions[1].coefficients.as_ref().unwrap();
    let mixed_high: f64 = gains.iter().map(|(w, _, high)| w * high).sum();
    close(high[0], mixed_high, 1e-9);
    close(high[1], -1.5 * 0.8, 1e-9);

    // Every identified structure's own response is retained; the masses are the compiler's.
    let mixture = executed.mixture.as_ref().unwrap();
    assert_eq!(mixture.atoms.iter().filter(|atom| atom.response.is_some()).count(), 2);
    close(mixture.identified_mass, 0.8, 1e-12);
    close(mixture.unidentified_mass, 0.2, 1e-12);
    let pooled = study_of_frequentist(&data).run(&ctx).unwrap();
    let pooled = pooled.structural_response.as_ref().expect("the compiler's own mixture");
    close(mixture.unidentified_mass, pooled.unidentified_mass, 1e-9);
    close(mixture.identified_mass, pooled.identified_mass, 1e-9);

    // The result-level functions on the retained structures give the same evaluation.
    let claims = pd::structure_claims(
        mixture,
        &[],
        data.schema(),
        &compiled.response_context(0, &ctx).unwrap(),
        StructuralPolicy::BayesOverStructures,
    )
    .unwrap();
    let by_hand = pd::decide_across_structures(&bayes_contract, &claims, None, options()).unwrap();
    assert_eq!(executed.evaluation.structures(), Some(&by_hand));

    // Invariance across structures needs every structure; the unidentified one blocks it.
    let invariant = contract(0.1, 0.5 * switch, StructuralPolicy::RequireInvariantBestAction);
    let executed = study
        .compile_for_preference_decision(&invariant, &ctx)
        .unwrap()
        .execute(None, options(), &ctx)
        .unwrap();
    let Some(StructuralPreferenceEvaluation::Exact(result)) = executed.evaluation.structures()
    else {
        panic!("an affine expectation is answered exactly");
    };
    assert!(matches!(result.verdict, StructuralPreferenceVerdict::InsufficientScience(_)));
    close(result.unidentified_mass.unwrap(), 0.2, 1e-9);
}

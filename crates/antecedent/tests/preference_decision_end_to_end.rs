//! Decisions across a range of preferences, from analyses run through the study surface.
//!
//! Fixture. A dose table `y = 0.5 + 1.5 t + noise` on `t` in `[0, 2]` with the graph `t -> y`,
//! and a mean response curve of `y` under `do(t = 0.5)` and `do(t = 1.5)`. The two actions read
//! one dose's mean benefit (points) each and carry a known cost (usd): `low` costs 0.5 and
//! `high` costs 1.5. The value is `benefit - price * cost` with the price (points per usd) a
//! declared range. With `gain = m(1.5) - m(0.5)` read from the analysis, `high` leads exactly
//! when `price < gain` (the costs differ by one), so a price range below `gain` makes `high`
//! best everywhere and a range around `gain` leaves the choice to the price, switching at
//! `gain`.
//!
//! Joint draws. A Bayesian analysis of the same table retains aligned posterior draws of both
//! means. A kinked model also charges a weighted shortfall `max(floor - benefit, 0)` with the
//! floor at the low dose's posterior mean benefit: its expectation over the draws is positive
//! while the shortfall of the mean is zero, so the exact route must read the draws; a quantile
//! criterion is not affine in the parameters and needs the certified route.
//!
//! Structures. A confounded table `y = 0.5 + 1.5 t + z + noise`, `t = 1 + 0.4 z + noise`, and a
//! posterior over three structures with weights `(0.5, 0.3, 0.2)`: `t -> y <- z`,
//! `z -> t -> y <- z`, and a cyclic `t <-> y` pair under which the effect is not identified.
//! Bayes over structures mixes the two identified structures' coefficients with weights 0.5
//! and 0.3 without renormalizing and reports the 0.2 unidentified mass.
//!
//! Learning. Recorded choices whose option beliefs are read from the analysis: choosing `low`
//! at costs `(0.5, 2.5)` gives `price >= (gain - tol) / 2`, choosing `high` at `(0.5, 1.75)`
//! gives `price <= (gain + tol) / 1.25`, which lies below `gain`, so the learned set settles
//! the decision the declared range `[0.2, 3]` leaves to the price. Noisy choices symmetric about
//! a cost difference of 1.75 give the logit estimate `price = gain / 1.75` exactly.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::native_claims::{
    NativeDecisionInput, NativeDecisionSource, NativeResponseClaim, NativeResponseContext,
};
use antecedent::analysis::preference_decision::{
    self as pd, AcceptedPreferences, ActionInput, ActionKind, Attribute, ChoiceModelOptions,
    ChoiceRecord, ChoiceRecordSet, DecisionCriterion, OptionBelief, PendingReview, Pooling,
    PreferenceAction, PreferenceAtomStatus, PreferenceBasis, PreferenceContract,
    PreferenceContractParts, PreferenceEvaluation, PreferenceEvaluationOptions, PreferenceExpr,
    PreferenceParameter, PreferencePolicy, PreferenceRoute, PreferenceSet, PreferenceVerdict,
    SetLearningOptions, StructuralPolicy, StructuralPreferenceEvaluation,
    StructuralPreferenceVerdict, ValueModel, shape,
};
use antecedent::design::decision_contract::{SourceRepresentation, SourceRequirement};
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_contract::{ProgramBinding, ScientificQuantity};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_discovery::adjacency_mask::{AdjacencyMask, set_edge};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::RetainedArtifactIdentity;
use antecedent_io::distribution_artifact::{DistributionArtifact, DistributionCalibration};
use antecedent_prob::InferenceDiagnostics;

#[allow(dead_code, reason = "each test file uses only some of the hand computations")]
#[path = "common/preference_kernels.rs"]
mod kernels;

const LOW_DOSE: f64 = 0.5;
const HIGH_DOSE: f64 = 1.5;
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

/// Deterministic noise in `[-1, 1]`.
fn wobble(i: u32, rate: f64) -> f64 {
    (f64::from(i) * rate).sin()
}

fn dose_table() -> TabularData {
    let n = 240_u32;
    let t: Vec<f64> = (0..n).map(|i| 2.0 * f64::from(i) / f64::from(n - 1)).collect();
    let y: Vec<f64> =
        (0..n).zip(&t).map(|(i, dose)| 0.5 + 1.5 * dose + 0.8 * wobble(i, 1.7)).collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())]).unwrap()
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

fn dag(variables: u32, edges: &[(u32, u32)]) -> Dag {
    let mut graph = Dag::with_variables(variables);
    for &(from, to) in edges {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn curve_query() -> CausalQuery {
    CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(Y),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(T),
            GridSpec::Values(Arc::from([LOW_DOSE, HIGH_DOSE])),
        ),
    }))
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64))
}

fn analyze(data: &TabularData, graph: Dag, inference: InferenceMode, seed: u64) -> StudyResult {
    Study::tabular(data.clone())
        .graph(graph)
        .query(curve_query())
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(seed))
        .unwrap()
}

fn context() -> NativeResponseContext {
    NativeResponseContext {
        program: ProgramBinding {
            graph_id: "graph:dose".into(),
            contract_id: "contract:dose".into(),
            treatment_id: "t".into(),
            outcome_id: "y".into(),
            population_id: "target".into(),
            intervention_kind: "do".into(),
            horizon: 0,
            dose_grid: vec![LOW_DOSE, HIGH_DOSE],
            dose_units: "mg".into(),
            outcome_units: "points".into(),
            functional_id: "mean".into(),
            transform_id: "identity".into(),
        },
        snapshot_id: "snapshot:dose-table".into(),
        rng_id: "pcg:seeded".into(),
        calibration: DistributionCalibration::NotMeasured,
    }
}

fn claim_of(result: &StudyResult, data: &TabularData) -> NativeResponseClaim {
    let response = result.response().expect("a mean response curve");
    NativeResponseClaim::from_response(response, data.schema(), &context()).unwrap()
}

/// `m(1.5) - m(0.5)` from the analysis: the price at which the two actions switch.
fn gain_of(claim: &NativeResponseClaim) -> f64 {
    claim.means()[1] - claim.means()[0]
}

fn attributes() -> Vec<Attribute> {
    vec![
        Attribute { id: "benefit".into(), units: "points".into() },
        Attribute { id: "cost".into(), units: "usd".into() },
    ]
}

fn price(lower: f64, upper: f64) -> PreferenceParameter {
    PreferenceParameter {
        id: "price".into(),
        units: "points/usd".into(),
        lower,
        upper,
        assumption_id: "assumption.price_of_benefit".into(),
    }
}

/// `benefit - price * cost`.
fn priced(lower: f64, upper: f64) -> ValueModel {
    ValueModel {
        attributes: attributes(),
        preferences: PreferenceSet::new(vec![price(lower, upper)], vec![]),
        value: PreferenceExpr::difference(
            PreferenceExpr::Attribute(0),
            PreferenceExpr::product(PreferenceExpr::Parameter(0), PreferenceExpr::Attribute(1)),
        ),
        utility_units: "points".into(),
    }
}

/// `benefit - price * cost - weight * max(floor - benefit, 0)`.
fn kinked(floor: f64, price_range: (f64, f64)) -> ValueModel {
    let mut model = priced(price_range.0, price_range.1);
    model.preferences.parameters.push(PreferenceParameter {
        id: "shortfall_weight".into(),
        units: "points/points".into(),
        lower: 0.0,
        upper: 2.0,
        assumption_id: "assumption.weight_of_shortfall".into(),
    });
    model.value = PreferenceExpr::difference(
        model.value,
        PreferenceExpr::product(PreferenceExpr::Parameter(1), shape::shortfall(0, floor)),
    );
    model
}

fn actions(claim: &NativeResponseClaim, costs: (f64, f64)) -> Vec<PreferenceAction> {
    let action = |id: &str, quantity: &ScientificQuantity, cost: f64| PreferenceAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![
            ActionInput::Quantity(quantity.clone()),
            ActionInput::Fixed { value: cost, units: "usd".into() },
        ],
    };
    let coordinates = claim.coordinates();
    vec![action("low", &coordinates[0], costs.0), action("high", &coordinates[1], costs.1)]
}

fn parts(
    claim: &NativeResponseClaim,
    criterion: DecisionCriterion,
    structural_policy: StructuralPolicy,
    preference_policy: PreferencePolicy,
) -> PreferenceContractParts {
    PreferenceContractParts {
        actions: actions(claim, (0.5, 1.5)),
        criterion,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy,
        preference_policy,
    }
}

fn contract(model: ValueModel, claim: &NativeResponseClaim) -> PreferenceContract {
    contract_with(model, claim, DecisionCriterion::PosteriorExpectedUtility)
}

fn contract_with(
    model: ValueModel,
    claim: &NativeResponseClaim,
    criterion: DecisionCriterion,
) -> PreferenceContract {
    AcceptedPreferences::declared(model).unwrap().contract(parts(
        claim,
        criterion,
        StructuralPolicy::RequireInvariantBestAction,
        PreferencePolicy::RequireInvariantBestAction,
    ))
}

fn joint_law(input: &NativeDecisionInput) -> &DistributionArtifact {
    let NativeDecisionSource::JointLaw(law) = &input.source else {
        panic!("the decision read aligned joint draws");
    };
    law
}

/// Weighted mean of `f` over column `column` of an equally weighted law.
fn draw_mean(law: &DistributionArtifact, column: usize, f: impl Fn(f64) -> f64) -> f64 {
    let [n, width] = law.shape();
    let total: f64 = (0..n).map(|draw| f(law.draws()[draw * width + column])).sum();
    total / f64::from(u32::try_from(n).unwrap())
}

#[test]
fn a_price_range_decides_from_an_analysed_dose_response() {
    let data = dose_table();
    let claim = claim_of(&analyze(&data, dag(2, &[(T, Y)]), InferenceMode::Frequentist, 11), &data);
    let gain = gain_of(&claim);
    assert!(gain > 1.0 && gain < 2.0, "{gain}");

    // A price range below the switch: `high` is best at every admissible price.
    let below =
        kernels::decide_claim(&contract(priced(0.2, 0.8 * gain), &claim), &claim, None, options())
            .unwrap();
    assert_eq!(below.input.representation, SourceRepresentation::Mean);
    assert_eq!(below.evaluation.route(), PreferenceRoute::Exact);
    assert!(below.evaluation.is_necessarily_best());
    assert_eq!(below.evaluation.chosen_action(), Some("high"));

    // A range around the switch: the choice depends on the price and switches at `gain`.
    let around = kernels::decide_claim(
        &contract(priced(0.5 * gain, 1.5 * gain), &claim),
        &claim,
        None,
        options(),
    )
    .unwrap();
    let exact = around.evaluation.exact().unwrap();
    assert_eq!(
        exact.verdict,
        PreferenceVerdict::PreferenceDependent { possibly_best: vec!["low".into(), "high".into()] }
    );
    assert_eq!(around.evaluation.chosen_action(), None);
    let switch = &exact.switches[0];
    assert_eq!((switch.first.as_str(), switch.second.as_str()), ("low", "high"));
    // `v_low - v_high = -gain + price * (1.5 - 0.5)`.
    close(switch.margin_coefficients[0], -gain, 1e-9);
    close(switch.margin_coefficients[1], 1.0, 1e-9);
    close(switch.margin_range.0, -0.5 * gain, 1e-9);
    close(switch.margin_range.1, 0.5 * gain, 1e-9);
    assert!(switch.favours_first_at[0] >= gain - 1e-9, "{:?}", switch.favours_first_at);
    assert!(switch.favours_second_at[0] <= gain + 1e-9, "{:?}", switch.favours_second_at);
    assert!(around.evaluation.assumptions().iter().any(|a| a == "assumption.price_of_benefit"));
}

#[test]
fn joint_draws_carry_a_kinked_value_and_a_quantile_takes_the_certified_route() {
    let data = dose_table();
    let claim = claim_of(&analyze(&data, dag(2, &[(T, Y)]), bayes(), 23), &data);
    assert!(claim.has_joint_law());
    let gain = gain_of(&claim);
    let floor = claim.means()[0];
    let expectation = contract(kinked(floor, (0.2, 0.8 * gain)), &claim);

    // The shortfall part is not affine in the benefit: the exact route reads the draws.
    let decided = kernels::decide_claim(&expectation, &claim, None, options()).unwrap();
    assert_eq!(decided.input.representation, SourceRepresentation::JointDraws);
    assert_eq!(decided.evaluation.route(), PreferenceRoute::Exact);
    assert_eq!(decided.evaluation.chosen_action(), Some("high"));
    assert!(decided.evaluation.is_necessarily_best());
    let law = joint_law(&decided.input);
    let expected_shortfall = draw_mean(law, 0, |benefit| (floor - benefit).max(0.0));
    let shortfall_of_mean = (floor - draw_mean(law, 0, |benefit| benefit)).max(0.0);
    assert!(expected_shortfall > shortfall_of_mean + 1e-6, "{expected_shortfall}");
    let low = &decided.evaluation.exact().unwrap().actions[0];
    let coefficients = low.coefficients.as_ref().unwrap();
    close(coefficients[1], -0.5, 1e-12);
    close(coefficients[2], -expected_shortfall, 1e-9);

    // A quantile of the value is not affine in the parameters: certified subdivision.
    let quantile = contract_with(
        kinked(floor, (0.2, 0.8 * gain)),
        &claim,
        DecisionCriterion::Quantile { p: 0.1 },
    );
    assert_eq!(
        pd::source_requirement(&quantile).unwrap().any_of,
        [SourceRepresentation::JointDraws]
    );
    let certified = kernels::decide_claim(&quantile, &claim, None, options()).unwrap();
    assert_eq!(certified.evaluation.route(), PreferenceRoute::Certified);
    assert_eq!(certified.evaluation.chosen_action(), Some("high"));
    assert!(certified.evaluation.undecided_fraction().is_some());

    // Means alone cannot answer either contract.
    let means = claim
        .decision_source(&SourceRequirement {
            any_of: vec![SourceRepresentation::Mean],
            sampled_needs_error_receipt: false,
        })
        .unwrap();
    let refused = kernels::decide(&quantile, &means.source, None, options()).unwrap_err();
    assert_eq!(refused.code, "joint_law_required");
    assert!(kernels::decide(&expectation, &means.source, None, options()).is_err());
}

fn posterior_of(masks: &[AdjacencyMask], weights: &[f64]) -> GraphPosterior {
    GraphPosterior::from_atoms(
        3,
        weights.to_vec(),
        masks.to_vec(),
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

fn mask(edges: &[(u32, u32)]) -> AdjacencyMask {
    edges.iter().fold(AdjacencyMask::EMPTY, |mask, &(from, to)| {
        set_edge(mask, 3, usize::try_from(from).unwrap(), usize::try_from(to).unwrap(), true)
    })
}

/// Graph-posterior estimation recurses deeply; run the body on a large stack.
fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    antecedent::support::caller_thread()
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("spawn the structural body")
        .join()
        .expect("the structural body completed");
}

#[test]
fn structures_from_a_graph_posterior_keep_their_unidentified_mass() {
    on_large_stack(structures_body);
}

#[allow(
    clippy::too_many_lines,
    reason = "one structural story: the posterior, each structure's analysis, both policies"
)]
fn structures_body() {
    let data = confounded_table();
    let covariate = [(T, Y), (Z, Y)];
    let confounder = [(Z, T), (Z, Y), (T, Y)];
    let cyclic = [(T, Y), (Y, T)];
    let posterior =
        posterior_of(&[mask(&covariate), mask(&confounder), mask(&cyclic)], &[0.5, 0.3, 0.2]);
    let keys = posterior.graph_keys().clone();

    // The study over the posterior says which structures identify the curve, and their mass.
    let mixed = Study::tabular(data.clone())
        .graph_posterior(posterior)
        .query(curve_query())
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(31))
        .unwrap();
    let mixture = mixed.structural_response.as_ref().expect("a structural mixture");
    close(mixture.unidentified_mass, 0.2, 1e-9);

    // Each identified structure's own analysis of the same table supplies its joint law.
    let first = analyze(&data, dag(3, &covariate), bayes(), 37);
    let second = analyze(&data, dag(3, &confounder), bayes(), 41);
    let per_structure = [
        (keys[0], first.response().expect("a response")),
        (keys[1], second.response().expect("a response")),
    ];
    let claims = [claim_of(&first, &data), claim_of(&second, &data)];
    let joint = SourceRequirement {
        any_of: vec![SourceRepresentation::JointDraws],
        sampled_needs_error_receipt: false,
    };
    let laws: Vec<NativeDecisionInput> =
        claims.iter().map(|claim| claim.decision_source(&joint).unwrap()).collect();
    let mean = |s: usize, column: usize| draw_mean(joint_law(&laws[s]), column, |x| x);
    // Bayes over structures, not renormalized: the combined gain is `0.5 g1 + 0.3 g2`, and the
    // combined price weight is `0.8`, so `high` leads below `(0.5 g1 + 0.3 g2) / 0.8`.
    let mixed_gain = 0.5 * (mean(0, 1) - mean(0, 0)) + 0.3 * (mean(1, 1) - mean(1, 0));
    let switch = mixed_gain / 0.8;
    let model = priced(0.1, 0.5 * switch);

    let adapted = pd::structure_claims(
        mixture,
        &per_structure,
        data.schema(),
        &context(),
        StructuralPolicy::BayesOverStructures,
    )
    .unwrap();
    assert_eq!(adapted.atoms.len(), 3);
    let bayes_contract = AcceptedPreferences::declared(model.clone()).unwrap().contract(parts(
        &claims[0],
        DecisionCriterion::PosteriorExpectedUtility,
        StructuralPolicy::BayesOverStructures,
        PreferencePolicy::RequireInvariantBestAction,
    ));
    let StructuralPreferenceEvaluation::Exact(result) =
        kernels::decide_across_structures(&bayes_contract, &adapted, None, options()).unwrap()
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
    // The unidentified mass could change the answer, so it is not a choice; the statement
    // conditional on the evaluated structures still is.
    assert!(matches!(result.verdict, StructuralPreferenceVerdict::InsufficientScience(_)));
    let combined = result.combined.as_ref().unwrap();
    assert_eq!(combined.verdict, PreferenceVerdict::NecessarilyBest("high".into()));
    let high = combined.actions[1].coefficients.as_ref().unwrap();
    close(high[0], 0.5 * mean(0, 1) + 0.3 * mean(1, 1), 1e-9);
    close(high[1], -1.5 * 0.8, 1e-9);

    // Invariance across structures needs every structure; the unidentified one blocks it.
    let invariant = AcceptedPreferences::declared(model).unwrap().contract(parts(
        &claims[0],
        DecisionCriterion::PosteriorExpectedUtility,
        StructuralPolicy::RequireInvariantBestAction,
        PreferencePolicy::RequireInvariantBestAction,
    ));
    let adapted = pd::structure_claims(
        mixture,
        &per_structure,
        data.schema(),
        &context(),
        StructuralPolicy::RequireInvariantBestAction,
    )
    .unwrap();
    let StructuralPreferenceEvaluation::Exact(result) =
        kernels::decide_across_structures(&invariant, &adapted, None, options()).unwrap()
    else {
        panic!("an affine expectation is answered exactly");
    };
    assert!(matches!(result.verdict, StructuralPreferenceVerdict::InsufficientScience(_)));
    close(result.unidentified_mass.unwrap(), 0.2, 1e-9);

    // A response for a structure the posterior does not hold is refused.
    let stray = [(keys[2], first.response().expect("a response"))];
    assert!(
        pd::structure_claims(
            mixture,
            &stray,
            data.schema(),
            &context(),
            StructuralPolicy::BayesOverStructures
        )
        .is_err()
    );
}

/// A recorded choice between the two doses at the given costs, with option beliefs read from
/// the analysis input.
fn record(
    id: &str,
    costs: (f64, f64),
    chosen: &str,
    claim: &NativeResponseClaim,
    input: &NativeDecisionInput,
) -> ChoiceRecord {
    let offered = AcceptedPreferences::declared(priced(0.2, 3.0)).unwrap().contract(
        PreferenceContractParts {
            actions: actions(claim, costs),
            ..parts(
                claim,
                DecisionCriterion::PosteriorExpectedUtility,
                StructuralPolicy::RequireInvariantBestAction,
                PreferencePolicy::ReportOnly,
            )
        },
    );
    ChoiceRecord {
        id: id.into(),
        decider: "pharmacy".into(),
        options: pd::analysis_options(&offered, &input.source).unwrap(),
        chosen: chosen.into(),
        belief_source: pd::analysis_belief_source(input, "2026-06-30"),
        observation_rows: vec![format!("ledger:{id}")],
    }
}

fn record_set(records: Vec<ChoiceRecord>) -> ChoiceRecordSet {
    ChoiceRecordSet {
        attributes: attributes(),
        population: "target".into(),
        records,
        pooling: Pooling::Pooled { assumption_id: "assumption.one_purchasing_policy".into() },
    }
}

/// The frequentist analysis, its mean input and the two bracketing records.
fn bracketing_records() -> (NativeResponseClaim, NativeDecisionInput, ChoiceRecordSet) {
    let data = dose_table();
    let claim = claim_of(&analyze(&data, dag(2, &[(T, Y)]), InferenceMode::Frequentist, 11), &data);
    let input = claim
        .decision_source(&pd::source_requirement(&contract(priced(0.2, 3.0), &claim)).unwrap())
        .unwrap();
    let records = record_set(vec![
        record("r1", (0.5, 2.5), "low", &claim, &input),
        record("r2", (0.5, 1.75), "high", &claim, &input),
    ]);
    (claim, input, records)
}

/// `analysis_result_identity`: a mean source is identified by whether its means are posterior
/// means or point estimates, and by its coordinates and means in order (each mean belongs to
/// the coordinate at its position).
#[test]
fn the_analysis_result_identity_names_the_estimate_kind_and_keeps_coordinate_order() {
    use antecedent_design::decision_eval::{MeanEstimate, MeanSource};
    let (_, input, _) = bracketing_records();
    let base = pd::analysis_result_identity(&input);
    let with = |edit: fn(&mut MeanSource)| {
        let mut changed = input.clone();
        let NativeDecisionSource::Mean(means) = &mut changed.source else {
            panic!("a frequentist claim gives a mean source");
        };
        edit(means);
        pd::analysis_result_identity(&changed)
    };
    assert_eq!(with(|_| {}), base);
    let kind = with(|m| {
        m.estimate = match m.estimate {
            MeanEstimate::Posterior => MeanEstimate::Frequentist,
            MeanEstimate::Frequentist => MeanEstimate::Posterior,
        };
    });
    assert_ne!(kind, base, "the estimate kind is part of the identity");
    let NativeDecisionSource::Mean(means) = &input.source else { unreachable!() };
    assert!(means.coordinates.len() >= 2, "the fixture reads several coordinates");
    let swapped = with(|m| {
        m.coordinates.swap(0, 1);
        m.means.swap(0, 1);
    });
    assert_ne!(swapped, base, "coordinates and means are ordered");
    let mut seen = std::collections::BTreeSet::from([base, kind, swapped]);
    for edit in [
        (|m: &mut MeanSource| m.means[0] += 1.0) as fn(&mut MeanSource),
        |m| m.provider_id.push('x'),
        |m| m.snapshot_id.push('x'),
        |m| m.causal_contract_id.push('x'),
        |m| m.rng_id.push('x'),
    ] {
        assert!(seen.insert(with(edit)));
    }
}

const TOLERANCE: f64 = 0.01;

fn reviewed(records: &ChoiceRecordSet) -> AcceptedPreferences {
    let learned = antecedent_design::accepted_preferences::learn_compatible_set_accepted(
        &priced(0.2, 3.0),
        records,
        None,
        SetLearningOptions { tolerance: TOLERANCE, ..SetLearningOptions::default() },
    )
    .unwrap();
    assert!(learned.pending().contains(&PendingReview::ToleranceApplied { tolerance: TOLERANCE }));
    learned.acknowledge(&learned.pending()).unwrap()
}

#[test]
fn recorded_choices_read_from_an_analysis_settle_a_price_dependent_decision() {
    let (claim, input, records) = bracketing_records();
    let gain = gain_of(&claim);
    assert_eq!(input.representation, SourceRepresentation::Mean);
    let first = &records.records[0];
    assert_eq!(
        first.belief_source,
        pd::BeliefSource::AnalysisAsOf {
            result_identity: pd::analysis_result_identity(&input),
            as_of: "2026-06-30".into(),
        }
    );
    let OptionBelief::Means(low) = &first.options[0].belief else {
        panic!("a mean source gives attribute means");
    };
    close(low[0], claim.means()[0], 1e-12);
    close(low[1], 0.5, 1e-12);

    let decision_parts = parts(
        &claim,
        DecisionCriterion::PosteriorExpectedUtility,
        StructuralPolicy::RequireInvariantBestAction,
        PreferencePolicy::RequireInvariantBestAction,
    );
    let declared = AcceptedPreferences::declared(priced(0.2, 3.0)).unwrap();
    let before =
        kernels::decide_accepted(&declared, decision_parts.clone(), &input.source, options())
            .unwrap();
    assert!(matches!(
        before.exact().unwrap().verdict,
        PreferenceVerdict::PreferenceDependent { .. }
    ));

    let accepted = reviewed(&records);
    assert_eq!(accepted.version(), 2);
    assert!(accepted.pending().is_empty());
    let PreferenceBasis::CompatibleSet(learned) = accepted.basis() else {
        panic!("the accepted basis is the compatible set");
    };
    let range = &learned.ranges[0];
    assert!(range.varied_in_records && range.narrowed);
    close(range.learned.0, (gain - TOLERANCE) / 2.0, 1e-6);
    close(range.learned.1, (gain + TOLERANCE) / 1.25, 1e-6);
    assert!(range.learned.1 < gain);

    let after =
        kernels::decide_accepted(&accepted, decision_parts, &input.source, options()).unwrap();
    assert_eq!(after.exact().unwrap().verdict, PreferenceVerdict::NecessarilyBest("high".into()));
}

#[test]
fn a_choice_estimate_feeds_the_measure_weighted_policy() {
    let (claim, input, _) = bracketing_records();
    let gain = gain_of(&claim);
    // Choices symmetric about a cost difference of 1.75: high, high at 0.5 and 1.0; high, low at
    // 1.5; low, high at 2.0; low, low at 2.5 and 3.0.
    let pattern = [
        (0.5, "high"),
        (0.5, "high"),
        (1.0, "high"),
        (1.0, "high"),
        (1.5, "high"),
        (1.5, "low"),
        (2.0, "low"),
        (2.0, "high"),
        (2.5, "low"),
        (2.5, "low"),
        (3.0, "low"),
        (3.0, "low"),
    ];
    let records = record_set(
        pattern
            .iter()
            .enumerate()
            .map(|(i, (difference, chosen))| {
                record(&format!("noisy-{i}"), (0.5, 0.5 + difference), chosen, &claim, &input)
            })
            .collect(),
    );
    let accepted = antecedent_design::accepted_preferences::estimate_accepted(
        &priced(0.2, 3.0),
        &records,
        None,
        ChoiceModelOptions { n_draws: 400, seed: 5, ..ChoiceModelOptions::default() },
    )
    .unwrap();
    let PreferenceBasis::ChoiceEstimate(estimate) = accepted.basis() else {
        panic!("the accepted basis is the estimate");
    };
    close(estimate.parameters[0].estimate, gain / 1.75, 1e-6);
    assert!(accepted.measure().is_some());

    let weighted = kernels::decide_accepted(
        &accepted,
        parts(
            &claim,
            DecisionCriterion::PosteriorExpectedUtility,
            StructuralPolicy::RequireInvariantBestAction,
            PreferencePolicy::Weighted,
        ),
        &input.source,
        options(),
    )
    .unwrap();
    let PreferenceVerdict::MeasureChoice { action, share } = &weighted.exact().unwrap().verdict
    else {
        panic!("the weighted policy chooses by the measure");
    };
    assert_eq!(action, "high");
    assert!(*share > 0.5, "{share}");
}

#[test]
fn accepted_preferences_and_decisions_are_exported_consumed_and_replayed() {
    let (_, _, records) = bracketing_records();
    let accepted = reviewed(&records);
    let bytes = pd::export_accepted(&accepted, "pharmacy-pricing").unwrap();
    let retained = RetainedArtifactIdentity::of_bytes(&bytes);
    assert_eq!(
        pd::consume_accepted(&bytes, accepted.identity(), Some(&records), &retained).unwrap(),
        accepted
    );
    let mut tampered = bytes.clone();
    let middle = tampered.len() / 2;
    tampered[middle] ^= 0x01;
    assert!(pd::consume_accepted(&tampered, accepted.identity(), None, &retained).is_err());
    let tampered_retained = RetainedArtifactIdentity::of_bytes(&tampered);
    assert!(
        pd::consume_accepted(&tampered, accepted.identity(), None, &tampered_retained).is_err()
    );
    assert!(pd::consume_accepted(&bytes, &"0".repeat(64), None, &retained).is_err());
    let fewer = ChoiceRecordSet { records: records.records[..1].to_vec(), ..records.clone() };
    assert!(pd::consume_accepted(&bytes, accepted.identity(), Some(&fewer), &retained).is_err());

    let data = dose_table();
    let claim = claim_of(&analyze(&data, dag(2, &[(T, Y)]), bayes(), 23), &data);
    let fewer_draws = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(48));
    let other = claim_of(&analyze(&data, dag(2, &[(T, Y)]), fewer_draws, 29), &data);
    let floor = claim.means()[0];
    let gain = gain_of(&claim);
    for criterion in
        [DecisionCriterion::PosteriorExpectedUtility, DecisionCriterion::Quantile { p: 0.1 }]
    {
        let decided = contract_with(kinked(floor, (0.2, 0.8 * gain)), &claim, criterion);
        let input = claim.decision_source(&pd::source_requirement(&decided).unwrap()).unwrap();
        let law = joint_law(&input);
        let artifact =
            pd::PreferenceDecisionArtifact::evaluate(decided, law, None, options()).unwrap();
        let bytes = pd::export_decision(&artifact, "dose-decision").unwrap();
        let retained = RetainedArtifactIdentity::of_bytes(&bytes);
        let back =
            pd::consume_decision(&bytes, artifact.contract_identity(), law, &retained).unwrap();
        assert_eq!(back.evaluation(), artifact.evaluation());
        assert_eq!(back.route(), artifact.route());
        assert!(matches!(
            back.evaluation(),
            PreferenceEvaluation::Exact(_) | PreferenceEvaluation::Certified(_)
        ));
        back.replay(law).unwrap();

        let mut tampered = bytes.clone();
        let middle = tampered.len() / 2;
        tampered[middle] ^= 0x01;
        assert!(
            pd::consume_decision(&tampered, artifact.contract_identity(), law, &retained).is_err()
        );
        let tampered_retained = RetainedArtifactIdentity::of_bytes(&tampered);
        assert!(
            pd::consume_decision(&tampered, artifact.contract_identity(), law, &tampered_retained)
                .is_err()
        );
        assert!(pd::consume_decision(&bytes, &"0".repeat(64), law, &retained).is_err());
        let elsewhere =
            other.decision_source(&pd::source_requirement(artifact.contract()).unwrap()).unwrap();
        assert!(
            pd::consume_decision(
                &bytes,
                artifact.contract_identity(),
                joint_law(&elsewhere),
                &retained
            )
            .is_err()
        );
    }
}

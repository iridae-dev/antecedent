//! A decision across preferences over graph-posterior class atoms weighed by a declared
//! measure over their members.
//!
//! Fixture. A confounded table over `t, y, z, w`: `z` follows `w`, `t = 1 + 0.4 z + noise` and
//! `y = 0.5 + 1.5 t + z + noise`, with a mean response curve of `y` under `do(t = 0.5)` and
//! `do(t = 1.5)`. The posterior holds two CPDAG atoms: the path `w - z - t` with `t -> y`
//! (three members: one per source of the path) at weight 0.6, and the fully oriented
//! `w -> z -> t -> y` at weight 0.4.
//!
//! * A uniform class prior makes every member a probability (atom weight over three), so
//!   `BayesOverStructures` runs; without a prior it stays refused.
//! * Keyed masses are read by member fingerprint and normalised within their class; a missing
//!   or extra key, ordered masses, and a sample under keyed masses are refused.
//! * Above a declared exact limit a uniform measure draws members: the draw is recorded, its
//!   seed reproduces it, the class mass is kept, and no verdict over every member is given
//!   (a Monte Carlo verdict on the sampled measure under Bayes, each sampled member's own
//!   verdict under the invariant policy).
//! * A PAG atom `z o-o t -> y` has three members; under `t -> z` the edge `t -> y` is
//!   invisible, so its member is unidentified and its third of the mass stays unidentified,
//!   never renormalized.
//! * Clopper–Pearson bounds match their closed forms at the ends.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::preference_decision::{
    self as pd, AcceptedPreferences, ActionInput, ActionKind, Attribute, ClassSamplingScheme,
    ClassSize, CompiledPreferenceDecision, CompiledPreferenceEvaluation, DecisionCriterion,
    PreferenceAction, PreferenceContract, PreferenceContractParts, PreferenceEvaluationOptions,
    PreferenceExpr, PreferenceParameter, PreferencePolicy, PreferenceSet, SampledStructuralVerdict,
    StructuralPolicy, StructuralPreferenceEvaluation, StructureIdentification, ValueModel,
};
use antecedent::reason::reason_details::preference_compile;
use antecedent::result::StructuralWeightBasis;
use antecedent::{BayesianConfig, ClassPrior, InferenceMode, RefuteSuite, Study};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_discovery::adjacency_mask::{AdjacencyMask, adjacency_masks_from_pag, set_edge};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_graph::{DenseNodeId, Pag};
use antecedent_prob::InferenceDiagnostics;

const T: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;
const W: u32 = 3;

const UNIFORM: &str = "assumption.uniform_within_class";

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

fn table() -> TabularData {
    let n = 300_u32;
    let w: Vec<f64> = (0..n).map(|i| wobble(i, 0.37)).collect();
    let z: Vec<f64> = (0..n).zip(&w).map(|(i, w)| 0.6 * w + 0.5 * wobble(i, 0.91)).collect();
    let t: Vec<f64> = (0..n).zip(&z).map(|(i, z)| 1.0 + 0.4 * z + 0.5 * wobble(i, 2.3)).collect();
    let y: Vec<f64> = (0..n)
        .zip(t.iter().zip(&z))
        .map(|(i, (t, z))| 0.5 + 1.5 * t + z + 0.3 * wobble(i, 1.3))
        .collect();
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("w", w.as_slice()),
    ])
    .unwrap()
}

/// The coded three-column table the mixed-graph route evaluates with its discrete plug-in.
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

/// Both directed bits of a pair encode an undirected CPDAG edge.
fn mask(n: usize, edges: &[(u32, u32)]) -> AdjacencyMask {
    edges.iter().fold(AdjacencyMask::EMPTY, |mask, &(from, to)| {
        set_edge(mask, n, usize::try_from(from).unwrap(), usize::try_from(to).unwrap(), true)
    })
}

fn summary() -> SamplingSummary {
    SamplingSummary {
        ess: 1.0,
        diagnostics: InferenceDiagnostics::analytic("preference_class_weighting"),
        rejected_invalid: 0,
    }
}

/// The path class `w - z - t` with `t -> y` at 0.6 and `w -> z -> t -> y` at 0.4.
fn cpdag_posterior() -> GraphPosterior {
    let path: &[(u32, u32)] = &[(W, Z), (Z, W), (Z, T), (T, Z), (T, Y)];
    let oriented: &[(u32, u32)] = &[(W, Z), (Z, T), (T, Y)];
    GraphPosterior::from_atoms(
        4,
        vec![0.6, 0.4],
        vec![mask(4, path), mask(4, oriented)],
        vec![0.0; 16],
        vec![0.0; 16],
        summary(),
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Cpdag)
}

/// The PAG `z o-o t -> y`.
fn pag_posterior() -> GraphPosterior {
    let mut pag = Pag::with_variables(3);
    pag.insert_circle_circle(DenseNodeId::from_raw(Z), DenseNodeId::from_raw(T)).unwrap();
    pag.insert_directed(DenseNodeId::from_raw(T), DenseNodeId::from_raw(Y)).unwrap();
    let (adjacency, marks) = adjacency_masks_from_pag(&pag).unwrap();
    GraphPosterior::from_atoms(3, vec![1.0], vec![adjacency], vec![0.0; 9], vec![0.0; 9], summary())
        .unwrap()
        .with_atom_kind(GraphPosteriorAtomKind::Pag)
        .with_mark_atoms([marks])
        .unwrap()
}

fn study(data: &TabularData, posterior: GraphPosterior, prior: Option<ClassPrior>) -> Study {
    let builder = Study::tabular(data.clone())
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
        .bootstrap_replicates(0);
    match prior {
        Some(prior) => builder.class_prior(prior),
        None => builder,
    }
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
fn contract(structural_policy: StructuralPolicy) -> PreferenceContract {
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
        .name("preference-class-weighting".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("spawn the structural body")
        .join()
        .expect("the structural body completed");
}

fn refusal_detail(error: &pd::PreferenceRouteError) -> (&str, &str) {
    let refusal = error.refusal().expect("a compile refusal");
    (refusal.code, &refusal.detail)
}

/// The members of the path class, in structure order.
fn path_members(compiled: &CompiledPreferenceDecision) -> Vec<&pd::CompiledStructure> {
    let path_key = cpdag_posterior().graph_keys()[0];
    compiled.structures().iter().filter(|s| s.class_key() == Some(path_key)).collect()
}

#[test]
fn a_uniform_prior_reads_class_members_as_probabilities() {
    on_large_stack(uniform_body);
}

fn uniform_body() {
    let data = table();
    let ctx = ExecutionContext::for_tests(41);
    let bayes = contract(StructuralPolicy::BayesOverStructures);

    // Without a declared measure Bayes over structures still refuses completion counts.
    let refused = study(&data, cpdag_posterior(), None)
        .compile_for_preference_decision(&bayes, &ctx)
        .unwrap_err();
    assert_eq!(
        refusal_detail(&refused).1,
        antecedent::reason::reason_details::decision_adapters::COMPLETION_COUNT_NOT_PROBABILITY
    );

    let uniform = ClassPrior::uniform(UNIFORM).unwrap();
    let structured = study(&data, cpdag_posterior(), Some(uniform));
    let compiled = structured.compile_for_preference_decision(&bayes, &ctx).unwrap();
    let members = path_members(&compiled);
    assert_eq!(members.len(), 3, "one member per source of the path");
    for member in &members {
        close(member.weight(), 0.6 / 3.0, 1e-12);
        let declared = member.member().expect("a class member");
        assert_eq!(declared.multiplicity, 1);
        assert!(declared.sampling.is_none(), "enumerated exactly");
        assert!(member.plan_view().is_some());
    }
    let prints: std::collections::BTreeSet<u64> =
        members.iter().filter_map(|m| m.fingerprint()).collect();
    assert_eq!(prints.len(), 3, "members have distinct content fingerprints");
    close(compiled.structures().iter().map(pd::CompiledStructure::weight).sum(), 1.0, 1e-12);
    assert!(compiled.class_sampling().is_empty());

    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let mixture = executed.mixture.as_ref().expect("a structural mixture");
    assert_eq!(mixture.weight_basis, StructuralWeightBasis::PosteriorWithDeclaredClassPrior);
    let Some(StructuralPreferenceEvaluation::Exact(result)) = executed.evaluation.structures()
    else {
        panic!("an exact evaluation across enumerated members");
    };
    let mut probabilities: Vec<f64> = result.atoms.iter().filter_map(|a| a.probability).collect();
    probabilities.sort_by(f64::total_cmp);
    assert_eq!(probabilities.len(), 4);
    for (probability, expected) in probabilities.iter().zip([0.2, 0.2, 0.2, 0.4]) {
        close(*probability, expected, 1e-9);
    }
    // The result-level functions on the retained structures give the same evaluation.
    let claims = pd::structure_claims(
        mixture,
        &[],
        data.schema(),
        &compiled.response_context(0, &ctx).unwrap(),
        StructuralPolicy::BayesOverStructures,
    )
    .unwrap();
    let by_hand = pd::decide_across_structures(&bayes, &claims, None, options()).unwrap();
    assert_eq!(executed.evaluation.structures(), Some(&by_hand));
}

#[test]
fn keyed_masses_are_read_by_member_fingerprint() {
    on_large_stack(keyed_body);
}

fn keyed_body() {
    let data = table();
    let ctx = ExecutionContext::for_tests(43);
    let invariant = contract(StructuralPolicy::RequireInvariantBestAction);
    let uniform = study(&data, cpdag_posterior(), Some(ClassPrior::uniform(UNIFORM).unwrap()))
        .compile_for_preference_decision(&invariant, &ctx)
        .unwrap();
    let mut prints: Vec<u64> =
        path_members(&uniform).iter().filter_map(|m| m.fingerprint()).collect();
    let oriented_print = uniform
        .structures()
        .iter()
        .find(|s| s.class_key() == Some(cpdag_posterior().graph_keys()[1]))
        .and_then(pd::CompiledStructure::fingerprint)
        .expect("the oriented atom is a one-member class");
    prints.sort_unstable();

    // Masses 1 : 2 : 1 over the path members (by sorted fingerprint), any mass on the
    // oriented atom's only member.
    let keyed = ClassPrior::from_pairs(vec![
        (prints[0], 1.0),
        (prints[1], 2.0),
        (prints[2], 1.0),
        (oriented_print, 5.0),
    ])
    .unwrap()
    .with_assumption("assumption.declared_member_masses")
    .unwrap();
    let compiled = study(&data, cpdag_posterior(), Some(keyed))
        .compile_for_preference_decision(&invariant, &ctx)
        .unwrap();
    for member in path_members(&compiled) {
        let print = member.fingerprint().unwrap();
        let expected = if print == prints[1] { 0.6 * 0.5 } else { 0.6 * 0.25 };
        close(member.weight(), expected, 1e-12);
    }

    // A member without a key is refused; no mass is invented for it.
    let missing = ClassPrior::from_pairs(vec![(prints[0], 1.0), (prints[1], 1.0)]).unwrap();
    let refused = study(&data, cpdag_posterior(), Some(missing))
        .compile_for_preference_decision(&invariant, &ctx)
        .unwrap_err();
    assert_eq!(
        refusal_detail(&refused),
        ("invalid_argument", preference_compile::CLASS_PRIOR_KEYS_MISMATCH)
    );
    // A key that names no member is refused.
    let mut extra: Vec<(u64, f64)> = prints.iter().map(|p| (*p, 1.0)).collect();
    extra.push((oriented_print, 1.0));
    extra.push((oriented_print ^ 1, 1.0));
    let refused = study(&data, cpdag_posterior(), Some(ClassPrior::from_pairs(extra).unwrap()))
        .compile_for_preference_decision(&invariant, &ctx)
        .unwrap_err();
    assert_eq!(refusal_detail(&refused).1, preference_compile::CLASS_PRIOR_KEYS_MISMATCH);
    // Ordered masses name no member of a graph-posterior class.
    let ordered = ClassPrior::from_ordered(vec![1.0, 1.0, 1.0]).unwrap();
    let refused = study(&data, cpdag_posterior(), Some(ordered))
        .compile_for_preference_decision(&invariant, &ctx)
        .unwrap_err();
    assert_eq!(refusal_detail(&refused).1, preference_compile::CLASS_PRIOR_KEYS_MISMATCH);
    // Keyed masses cannot be sampled: a sample needs a uniform measure.
    let sampled = ClassPrior::from_pairs(vec![(prints[0], 1.0)]).unwrap().sample(10, 1).unwrap();
    let refused = study(&data, cpdag_posterior(), Some(sampled))
        .compile_for_preference_decision(&invariant, &ctx)
        .unwrap_err();
    assert_eq!(
        refusal_detail(&refused),
        ("invalid_argument", preference_compile::CLASS_PRIOR_SAMPLING_NEEDS_UNIFORM)
    );
}

#[test]
fn a_class_above_the_exact_limit_is_sampled_and_never_called_best_everywhere() {
    on_large_stack(sampled_body);
}

fn sampled_prior(seed: u64) -> ClassPrior {
    ClassPrior::uniform(UNIFORM).unwrap().exact_limit(2).unwrap().sample(40, seed).unwrap()
}

fn sampled_body() {
    let data = table();
    let ctx = ExecutionContext::for_tests(47);
    let path_key = cpdag_posterior().graph_keys()[0];

    // Above the limit without a sample the class is refused, naming the narrowed cause.
    let limited = ClassPrior::uniform(UNIFORM).unwrap().exact_limit(2).unwrap();
    let refused = study(&data, cpdag_posterior(), Some(limited))
        .compile_for_preference_decision(&contract(StructuralPolicy::BayesOverStructures), &ctx)
        .unwrap_err();
    assert_eq!(
        refusal_detail(&refused),
        ("route_not_supported", preference_compile::CLASS_COMPLETIONS_TRUNCATED)
    );

    let bayes = contract(StructuralPolicy::BayesOverStructures);
    let compiled = study(&data, cpdag_posterior(), Some(sampled_prior(9)))
        .compile_for_preference_decision(&bayes, &ctx)
        .unwrap();
    let sampling = compiled.class_sampling();
    assert_eq!(sampling.len(), 1, "only the path class exceeds the limit");
    let record = &sampling[0];
    assert_eq!(record.class_key, path_key);
    assert_eq!(record.scheme, ClassSamplingScheme::CliquePicking);
    assert_eq!(record.class_size, ClassSize::Exact(3));
    assert_eq!((record.drawn, record.seed), (40, 9));
    assert_eq!(record.assumption.as_deref(), Some(UNIFORM));
    let members = path_members(&compiled);
    assert_eq!(members.len(), record.distinct);
    let drawn: usize = members.iter().map(|m| m.member().unwrap().multiplicity).sum();
    assert_eq!(drawn, 40);
    close(members.iter().map(|m| m.weight()).sum(), 0.6, 1e-12);
    for member in &members {
        let multiplicity = member.member().unwrap().multiplicity as f64;
        close(member.weight(), 0.6 * multiplicity / 40.0, 1e-12);
    }

    // The seed reproduces the draw; another seed records itself.
    let again = study(&data, cpdag_posterior(), Some(sampled_prior(9)))
        .compile_for_preference_decision(&bayes, &ctx)
        .unwrap();
    let keyed = |c: &CompiledPreferenceDecision| -> Vec<(Option<u64>, f64)> {
        c.structures().iter().map(|s| (s.key(), s.weight())).collect()
    };
    assert_eq!(keyed(&again), keyed(&compiled));
    let reseeded = study(&data, cpdag_posterior(), Some(sampled_prior(10)))
        .compile_for_preference_decision(&bayes, &ctx)
        .unwrap();
    assert_eq!(reseeded.class_sampling()[0].seed, 10);

    // Bayes over structures: a verdict on the sampled measure, with the class masses and the
    // Monte Carlo errors of the margins; never a verdict over every member.
    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let CompiledPreferenceEvaluation::Sampled(sampled) = &executed.evaluation else {
        panic!("a sampled class is decided on the sampled measure");
    };
    assert!(executed.evaluation.structures().is_none());
    assert!(matches!(sampled.verdict, SampledStructuralVerdict::OnSampledMeasure { .. }));
    assert_eq!(sampled.sampling.len(), 1);
    close(sampled.level, pd::SAMPLED_MASS_LEVEL, 1e-15);
    let [mass] = sampled.masses.as_slice() else {
        panic!("one sampled class");
    };
    close(mass.class_mass, 0.6, 1e-12);
    close(
        mass.identified.estimate + mass.unidentified.estimate + mass.unevaluated.estimate,
        0.6,
        1e-12,
    );
    assert!(mass.identified.lower <= mass.identified.estimate);
    assert!(mass.identified.estimate <= mass.identified.upper);
    assert!(mass.identified.upper <= 0.6 + 1e-12);
    assert!(!sampled.margins.is_empty(), "the exact route reports margins at the vertices");
    for margin in &sampled.margins {
        assert_eq!((margin.first.as_str(), margin.second.as_str()), ("low", "high"));
        assert!(margin.standard_error.is_some_and(|se| se.is_finite() && se >= 0.0));
    }
    assert!(sampled.statement().contains("sampled"));

    // The invariant policy needs every member: each sampled member is evaluated on its own and
    // no combined verdict is given.
    let invariant = contract(StructuralPolicy::RequireInvariantBestAction);
    let compiled = study(&data, cpdag_posterior(), Some(sampled_prior(9)))
        .compile_for_preference_decision(&invariant, &ctx)
        .unwrap();
    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let CompiledPreferenceEvaluation::Sampled(sampled) = &executed.evaluation else {
        panic!("a sampled class is reported per member");
    };
    let StructuralPreferenceEvaluation::Exact(result) = &sampled.evaluation else {
        panic!("an exact per-structure evaluation");
    };
    assert_eq!(result.structural_policy, StructuralPolicy::ReportOnly);
    let SampledStructuralVerdict::PerSampledMember { best_on_every_sampled_member, other_best } =
        &sampled.verdict
    else {
        panic!("no combined verdict from a sample under the invariant policy");
    };
    if best_on_every_sampled_member.is_some() {
        let [bound] = other_best.as_slice() else {
            panic!("one bound per sampled class");
        };
        assert_eq!((bound.disagreeing, bound.drawn), (0, 40));
        close(bound.upper, 0.6 * (1.0 - 0.05_f64.powf(1.0 / 40.0)), 1e-9);
    }
}

#[test]
fn a_pag_class_keeps_its_unidentified_member_mass() {
    on_large_stack(pag_body);
}

fn pag_body() {
    let data = coded_table();
    let ctx = ExecutionContext::for_tests(53);
    let bayes = contract(StructuralPolicy::BayesOverStructures);
    let compiled = study(&data, pag_posterior(), Some(ClassPrior::uniform(UNIFORM).unwrap()))
        .compile_for_preference_decision(&bayes, &ctx)
        .unwrap();
    let structures = compiled.structures();
    assert_eq!(structures.len(), 3, "z -> t, t -> z and z <-> t, each with t -> y");
    for structure in structures {
        close(structure.weight(), 1.0 / 3.0, 1e-12);
    }
    let unidentified = structures
        .iter()
        .filter(|s| matches!(s.identification(), StructureIdentification::NotIdentified(_)))
        .count();
    assert_eq!(unidentified, 1, "an invisible t -> y under t -> z is not identified");
    close(compiled.unidentified_mass(), 1.0 / 3.0, 1e-12);

    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let mixture = executed.mixture.as_ref().expect("a structural mixture");
    close(mixture.unidentified_mass, 1.0 / 3.0, 1e-12);
    close(mixture.identified_mass + mixture.unevaluable_mass, 2.0 / 3.0, 1e-12);
    let Some(StructuralPreferenceEvaluation::Exact(result)) = executed.evaluation.structures()
    else {
        panic!("an exact evaluation across enumerated members");
    };
    close(result.unidentified_mass.expect("probabilities were declared"), 1.0 / 3.0, 1e-9);

    // Sampled under the same measure: the draw is uniform over the three MAGs, and the masses
    // of the draws are estimated, never renormalized.
    let prior =
        ClassPrior::uniform(UNIFORM).unwrap().exact_limit(2).unwrap().sample(30, 4).unwrap();
    let compiled = study(&data, pag_posterior(), Some(prior))
        .compile_for_preference_decision(&bayes, &ctx)
        .unwrap();
    let sampling = compiled.class_sampling();
    let [record] = sampling.as_slice() else {
        panic!("one sampled class");
    };
    assert_eq!(record.scheme, ClassSamplingScheme::CircleEndpointRejection);
    assert_eq!(record.class_size, ClassSize::Exact(3));
    assert!(record.attempts.is_some_and(|attempts| attempts >= 30));
    let executed = compiled.execute(None, options(), &ctx).unwrap();
    let CompiledPreferenceEvaluation::Sampled(sampled) = &executed.evaluation else {
        panic!("a sampled class is decided on the sampled measure");
    };
    let [mass] = sampled.masses.as_slice() else {
        panic!("one sampled class");
    };
    close(
        mass.identified.estimate + mass.unidentified.estimate + mass.unevaluated.estimate,
        1.0,
        1e-12,
    );
    close(mass.unidentified.estimate, executed.mixture.as_ref().unwrap().unidentified_mass, 1e-12);
}

#[test]
fn a_pag_class_the_sampling_budget_cannot_audit_is_refused() {
    on_large_stack(budget_body);
}

fn budget_body() {
    // Nineteen nodes, three of them joined by circle-circle edges `v0 o-o v1 o-o v2` (five
    // members, above an exact limit of two): one m-separation model of nineteen nodes does not
    // fit the sampling budget, so no draw can be audited for membership.
    let n = 19_u32;
    let mut pag = Pag::with_variables(n);
    for i in 0..2 {
        pag.insert_circle_circle(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1)).unwrap();
    }
    let (adjacency, marks) = adjacency_masks_from_pag(&pag).unwrap();
    let cells = usize::try_from(n * n).unwrap();
    let posterior = GraphPosterior::from_atoms(
        19,
        vec![1.0],
        vec![adjacency],
        vec![0.0; cells],
        vec![0.0; cells],
        summary(),
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Pag)
    .with_mark_atoms([marks])
    .unwrap();
    let rows = 120_u32;
    let names: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
    let columns: Vec<Vec<f64>> =
        (0..n).map(|j| (0..rows).map(|i| wobble(i, 0.3 + f64::from(j) * 0.11)).collect()).collect();
    let pairs: Vec<(&str, &[f64])> =
        names.iter().map(String::as_str).zip(columns.iter().map(Vec::as_slice)).collect();
    let data = TabularData::from_f64_columns(pairs).unwrap();
    let prior =
        ClassPrior::uniform(UNIFORM).unwrap().exact_limit(2).unwrap().sample(50, 1).unwrap();
    let mut bayes = contract(StructuralPolicy::BayesOverStructures);
    for action in &mut bayes.actions {
        if let ActionInput::Quantity(quantity) = &mut action.inputs[0] {
            quantity.variable_id = "v1".into();
            quantity.variable_name = "v1".into();
            quantity.regime_id = quantity.regime_id.replace("do(t=", "do(v0=");
        }
    }
    let refused = study(&data, posterior, Some(prior))
        .compile_for_preference_decision(&bayes, &ExecutionContext::for_tests(59))
        .unwrap_err();
    assert_eq!(
        refusal_detail(&refused),
        ("route_not_supported", preference_compile::CLASS_SAMPLING_BUDGET)
    );
}

#[test]
fn clopper_pearson_bounds_match_their_closed_forms() {
    for n in [1_u64, 7, 40, 500] {
        let nf = n as f64;
        close(pd::clopper_pearson_upper(0, n, 0.95).unwrap(), 1.0 - 0.05_f64.powf(1.0 / nf), 1e-9);
        let (lower, upper) = pd::clopper_pearson(0, n, 0.95).unwrap();
        close(lower, 0.0, 1e-15);
        close(upper, 1.0 - 0.025_f64.powf(1.0 / nf), 1e-9);
        let (lower, upper) = pd::clopper_pearson(n, n, 0.95).unwrap();
        close(lower, 0.025_f64.powf(1.0 / nf), 1e-9);
        close(upper, 1.0, 1e-15);
    }
    // The interval of x successes mirrors that of n - x failures.
    let (lower, upper) = pd::clopper_pearson(7, 40, 0.9).unwrap();
    let (mirror_lower, mirror_upper) = pd::clopper_pearson(33, 40, 0.9).unwrap();
    close(lower, 1.0 - mirror_upper, 1e-9);
    close(upper, 1.0 - mirror_lower, 1e-9);
    assert!(pd::clopper_pearson(1, 0, 0.95).is_none());
    assert!(pd::clopper_pearson(3, 2, 0.95).is_none());
    assert!(pd::clopper_pearson_upper(0, 10, 1.0).is_none());
}

#[test]
fn the_graph_posterior_mixture_refuses_a_class_prior_it_does_not_read() {
    on_large_stack(unread_prior_body);
}

fn unread_prior_body() {
    // The mixture estimates each class atom as a whole; a measure over members is read only
    // by a decision compiled across the posterior's structures, so the estimate refuses it.
    let data = table();
    let ctx = ExecutionContext::for_tests(43);
    let prepared = study(&data, cpdag_posterior(), Some(ClassPrior::uniform(UNIFORM).unwrap()))
        .prepare(&ctx)
        .unwrap();
    let refused = prepared.estimate_retained(&ctx).unwrap_err().to_string();
    assert!(
        refused.contains(
            antecedent::reason::reason_details::structure_posterior::CLASS_PRIOR_NOT_READ
        ),
        "{refused}"
    );
    assert!(refused.contains("option_not_applicable"), "{refused}");
    // Without the prior the same mixture estimates.
    let plain = study(&data, cpdag_posterior(), None).prepare(&ctx).unwrap();
    assert!(plain.estimate_retained(&ctx).is_ok());
}

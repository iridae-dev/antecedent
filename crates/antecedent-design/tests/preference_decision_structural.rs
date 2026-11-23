//! Exact verdicts across structures and a preference set: every structural policy on two or
//! three structures, unidentified and unevaluated structures, structure probabilities and
//! refusals, and a brute-force cross-check of every linear-program quantity against a dense
//! grid over the preference set.
//!
//! The hand cases use the value `benefit - w * cost`; each structure fixes `(benefit, cost)`
//! per action, so `v_a(s, w) = benefit - w * cost`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_design::decision_contract::{
    ActionKind, DecisionCriterion, HardConstraint, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_structural::{AtomEvidence, StructuralAtom};
use antecedent_design::preference_contract::{
    ActionInput, PreferenceAction, PreferenceContract, PreferencePolicy,
};
use antecedent_design::preference_decision::{
    AtomChoice, CertifiedRouteReason, MeasureOrigin, PreferenceAtomStatus,
    PreferenceDecisionError as E, PreferenceDecisionOptions, PreferenceDecisionResult,
    PreferenceMeasure, PreferenceSwitch, PreferenceVerdict, StructuralPreferenceResult,
    StructuralPreferenceVerdict, evaluate_preference_structural,
};
use antecedent_design::preference_model::{
    Attribute, LinearRelation, PreferenceExpr as P, PreferenceParameter, PreferenceSet,
    RelationOrigin, RelationSense, ValueModel, shape,
};
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::quantity_wire::DistributionMeaningWire;

mod common;

const INVARIANCE_UNCHECKED: &str =
    "an unidentified or unevaluated structure leaves invariance unchecked";
const WORST_CASE_NEEDS_EVERY_STRUCTURE: &str =
    "a worst case over structures needs every structure evaluated";
const UNRESOLVED_MASS_CAN_CHANGE_BAYES: &str =
    "positive unresolved probability mass can change the Bayes action";

fn quantity(variable: &str, regime: &str, units: &str) -> ScientificQuantity {
    ScientificQuantity {
        units: units.into(),
        regime_id: regime.into(),
        ..common::quantity(variable)
    }
}

fn regime(action: usize) -> String {
    format!("do(a={action})")
}

fn parameter(id: &str, lower: f64, upper: f64) -> PreferenceParameter {
    PreferenceParameter {
        id: id.into(),
        units: "utility/unit".into(),
        lower,
        upper,
        assumption_id: format!("{id}-judgment"),
    }
}

/// `benefit - w * cost`, `w` in `[lower, upper]`.
fn model(lower: f64, upper: f64) -> ValueModel {
    ValueModel {
        attributes: vec![
            Attribute { id: "benefit".into(), units: "events".into() },
            Attribute { id: "cost".into(), units: "usd".into() },
        ],
        preferences: PreferenceSet {
            parameters: vec![parameter("w", lower, upper)],
            relations: vec![],
        },
        value: P::difference(shape::rate(0), P::product(P::Parameter(0), shape::rate(1))),
        utility_units: "events".into(),
    }
}

fn contract_for(
    model: ValueModel,
    n_actions: usize,
    structural: StructuralPolicy,
    preference: PreferencePolicy,
) -> PreferenceContract {
    let actions: Vec<PreferenceAction> = (0..n_actions)
        .map(|a| PreferenceAction {
            id: ["A", "B", "C", "D"][a].into(),
            kind: ActionKind::Intervention,
            inputs: model
                .attributes
                .iter()
                .map(|attribute| {
                    ActionInput::Quantity(quantity(&attribute.id, &regime(a), &attribute.units))
                })
                .collect(),
        })
        .collect();
    PreferenceContract {
        model,
        actions,
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: structural,
        preference_policy: preference,
    }
}

/// Joint equally weighted draws: `arms[a][k]` holds the draws of attribute `k` under action `a`.
fn law(model: &ValueModel, arms: &[Vec<Vec<f64>>]) -> Box<DistributionArtifact> {
    let mut columns = Vec::new();
    for a in 0..arms.len() {
        for attribute in &model.attributes {
            columns.push(quantity(&attribute.id, &regime(a), &attribute.units));
        }
    }
    let n = arms[0][0].len();
    let mut draws = Vec::new();
    for i in 0..n {
        for arm in arms {
            for attribute in arm {
                draws.push(attribute[i]);
            }
        }
    }
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "structure".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "enumeration".into(),
            causal_contract_id: "checked".into(),
        },
    )
    .unwrap();
    Box::new(
        DistributionArtifact::new(
            DistributionMetadata {
                version: 1,
                identity,
                axes: ["draw".into(), "quantity".into()],
                shape: [n, columns.len()],
                weights: None,
                supported: None,
                calibration: DistributionCalibration::Exact,
                trust: DistributionTrust::Unverified,
                legacy_posterior: None,
                legacy_bindings: None,
            },
            draws,
        )
        .unwrap(),
    )
}

/// One structure in which action `a` has `(benefit, cost) = values[a]` on two equal draws.
fn evaluated(id: &str, probability: Option<f64>, values: &[(f64, f64)]) -> StructuralAtom {
    let m = model(0.0, 1.0);
    let arms: Vec<Vec<Vec<f64>>> =
        values.iter().map(|(b, c)| vec![vec![*b, *b], vec![*c, *c]]).collect();
    atom(id, probability, AtomEvidence::Evaluated(law(&m, &arms)))
}

fn atom(id: &str, probability: Option<f64>, evidence: AtomEvidence) -> StructuralAtom {
    StructuralAtom { id: id.into(), probability, evidence }
}

/// A necessarily best on [0, 1]: margin 4.5 - 3w.
const S1: [(f64, f64); 2] = [(10.0, 4.0), (5.5, 1.0)];
/// A necessarily best on [0, 1]: margin 4 - w.
const S2: [(f64, f64); 2] = [(9.0, 2.0), (5.0, 1.0)];
/// A switch on [0, 1] at w = 1/6: margin 0.5 - 3w.
const S3: [(f64, f64); 2] = [(6.0, 4.0), (5.5, 1.0)];
/// With `S1` on `[0, 2]`, the worst case of A crosses: `min(10 - 4w, 7 - w)`.
const S4: [(f64, f64); 2] = [(7.0, 1.0), (6.0, 1.0)];

fn options() -> PreferenceDecisionOptions {
    PreferenceDecisionOptions::default()
}

fn run(
    lower: f64,
    upper: f64,
    structural: StructuralPolicy,
    preference: PreferencePolicy,
    atoms: &[StructuralAtom],
) -> StructuralPreferenceResult {
    let c = contract_for(model(lower, upper), 2, structural, preference);
    evaluate_preference_structural(&c, atoms, None, options()).unwrap()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
}

fn close_pair(got: Option<(f64, f64)>, want: (f64, f64)) -> bool {
    got.is_some_and(|(lo, hi)| close(lo, want.0) && close(hi, want.1))
}

fn atom_verdict(result: &StructuralPreferenceResult, index: usize) -> PreferenceVerdict {
    match &result.atoms[index].status {
        PreferenceAtomStatus::Evaluated(inner) => inner.verdict.clone(),
        other => panic!("structure {index} was not evaluated: {other:?}"),
    }
}

fn both() -> Vec<String> {
    vec!["A".into(), "B".into()]
}

#[test]
fn invariance_needs_every_structure_to_agree() {
    let invariant = StructuralPolicy::RequireInvariantBestAction;
    let policy = PreferencePolicy::RequireInvariantBestAction;
    let agree = [evaluated("s1", None, &S1), evaluated("s2", None, &S2)];
    let result = run(0.0, 1.0, invariant, policy, &agree);
    assert_eq!(
        result.verdict,
        StructuralPreferenceVerdict::Combined(PreferenceVerdict::NecessarilyBest("A".into()))
    );
    let combined = result.combined.as_ref().unwrap();
    // Ranges over structures: A in [6, 10] and [7, 9]; B in [4.5, 5.5] and [4, 5].
    assert!(close_pair(combined.actions[0].value_range, (6.0, 10.0)));
    assert!(close_pair(combined.actions[1].value_range, (4.0, 5.5)));
    assert!(combined.actions.iter().all(|o| o.coefficients.is_none()));
    assert!(close(combined.actions[1].worst_case_regret.unwrap(), 4.5));
    assert_eq!(combined.n_draws, 4);
    assert_eq!(combined.source.provider_id, "exact-law");
    assert_eq!((result.evaluated_mass, result.unidentified_mass), (None, None));

    let disagree = [evaluated("s1", None, &S1), evaluated("s3", None, &S3)];
    let result = run(0.0, 1.0, invariant, policy, &disagree);
    assert_eq!(
        result.verdict,
        StructuralPreferenceVerdict::StructuresDisagree(vec![
            AtomChoice {
                atom: "s1".into(),
                verdict: PreferenceVerdict::NecessarilyBest("A".into())
            },
            AtomChoice {
                atom: "s3".into(),
                verdict: PreferenceVerdict::PreferenceDependent { possibly_best: both() },
            },
        ])
    );
    let combined = result.combined.as_ref().unwrap();
    assert_eq!(combined.verdict, PreferenceVerdict::PreferenceDependent { possibly_best: both() });
    // The combined margin spans s3's low end and s1's high end.
    let switch = &combined.switches[0];
    assert!(close(switch.margin_range.0, -2.5) && close(switch.margin_range.1, 4.5));
    assert!(switch.margin_coefficients.is_empty());

    let unresolved = [evaluated("s1", None, &S1), atom("s2", None, AtomEvidence::Unidentified)];
    let result = run(0.0, 1.0, invariant, policy, &unresolved);
    assert_eq!(
        result.verdict,
        StructuralPreferenceVerdict::InsufficientScience(INVARIANCE_UNCHECKED.into())
    );
    assert_eq!(result.atoms[1].status, PreferenceAtomStatus::Unidentified);
    // A report-only preference policy claims nothing, so nothing is missing.
    let result = run(0.0, 1.0, invariant, PreferencePolicy::ReportOnly, &unresolved);
    assert_eq!(
        result.verdict,
        StructuralPreferenceVerdict::Combined(PreferenceVerdict::ReportOnly)
    );
}

/// A never beaten on [0, 1] but tied with B at w = 1: margin 3 - 3w.
const S6: [(f64, f64); 2] = [(10.0, 4.0), (7.0, 1.0)];

#[test]
fn an_action_unbeaten_in_every_structure_but_tied_in_one_is_never_beaten() {
    let invariant = StructuralPolicy::RequireInvariantBestAction;
    let policy = PreferencePolicy::RequireInvariantBestAction;
    let result = run(0.0, 1.0, invariant, policy, &[evaluated("s6", None, &S6)]);
    let tied = PreferenceVerdict::NeverBeaten { action: "A".into(), ties_with: vec!["B".into()] };
    assert_eq!(atom_verdict(&result, 0), tied);

    // Unique in s1, tied at an edge in s6: never beaten across structures, not unique.
    let result =
        run(0.0, 1.0, invariant, policy, &[evaluated("s1", None, &S1), evaluated("s6", None, &S6)]);
    assert_eq!(result.verdict, StructuralPreferenceVerdict::Combined(tied.clone()));
    let combined = result.combined.as_ref().unwrap();
    assert_eq!(combined.verdict, tied);
    let (a, b) = (&combined.actions[0], &combined.actions[1]);
    assert!(a.necessarily_best && !a.uniquely_best);
    assert!(!b.necessarily_best && b.dominated_by.is_empty());
    assert_eq!(
        result.statement(),
        "Action A is never beaten for any admissible value of the declared trade-offs and every \
         evaluated structure, and ties with B at some admissible values."
    );

    // Two structures where A is unique keep the unique winner across structures.
    let result =
        run(0.0, 1.0, invariant, policy, &[evaluated("s1", None, &S1), evaluated("s2", None, &S2)]);
    let combined = result.combined.as_ref().unwrap();
    assert!(combined.actions[0].uniquely_best);
    assert_eq!(combined.actions[1].dominated_by, vec!["A".to_string()]);
    assert_eq!(
        result.statement(),
        "Action A is preferred for every admissible value of the declared trade-offs and every \
         evaluated structure."
    );

    // Weighting the structures mixes the coefficients: A - B = 3.75 - 3w on [0, 1], unique.
    let bayes = run(
        0.0,
        1.0,
        StructuralPolicy::BayesOverStructures,
        policy,
        &[evaluated("s1", Some(0.5), &S1), evaluated("s6", Some(0.5), &S6)],
    );
    assert_eq!(
        bayes.verdict,
        StructuralPreferenceVerdict::Combined(PreferenceVerdict::NecessarilyBest("A".into()))
    );
    assert_eq!(
        bayes.statement(),
        "Action A is preferred for every admissible value of the declared trade-offs, with the \
         evaluated structures weighted by their probabilities."
    );
    // Disagreeing structures are named.
    let disagree =
        run(0.0, 1.0, invariant, policy, &[evaluated("s1", None, &S1), evaluated("s3", None, &S3)]);
    assert_eq!(
        disagree.statement(),
        "The evaluated structures s1 and s3 give different verdicts, so no action is preferred \
         across structures."
    );
}

#[test]
fn the_worst_case_over_structures_is_answered_by_linear_programs() {
    // On [0, 2]: g_A = min(10 - 4w, 7 - w) (7 - w up to w = 1, then 10 - 4w), g_B = 5.5 - w.
    // g_A - g_B is 1.5 on [0, 1] and 4.5 - 3w beyond, so it spans [-1.5, 1.5].
    let atoms = [evaluated("s1", None, &S1), evaluated("s4", None, &S4)];
    let worst = StructuralPolicy::Maximin;
    let result = run(0.0, 2.0, worst, PreferencePolicy::RequireInvariantBestAction, &atoms);
    assert_eq!(
        result.verdict,
        StructuralPreferenceVerdict::Combined(PreferenceVerdict::PreferenceDependent {
            possibly_best: both()
        })
    );
    let combined = result.combined.as_ref().unwrap();
    assert!(close_pair(combined.actions[0].value_range, (2.0, 7.0)));
    assert!(close_pair(combined.actions[1].value_range, (3.5, 5.5)));
    let switch = &combined.switches[0];
    assert!(close(switch.margin_range.0, -1.5) && close(switch.margin_range.1, 1.5));
    assert!(switch.margin_coefficients.is_empty());
    assert!(close(switch.favours_second_at[0], 2.0));
    assert!(switch.favours_first_at[0] <= 1.0 + 1e-9);
    assert_eq!(combined.draw_share_best_everywhere, None);

    let result = run(0.0, 2.0, worst, PreferencePolicy::Maximin, &atoms);
    assert_eq!(
        result.verdict,
        StructuralPreferenceVerdict::Combined(PreferenceVerdict::WorstCaseChoice("B".into()))
    );
    // Both worst-case regrets are 1.5.
    let result = run(0.0, 2.0, worst, PreferencePolicy::MinimaxRegret, &atoms);
    assert_eq!(
        result.verdict,
        StructuralPreferenceVerdict::Combined(PreferenceVerdict::Tied(both()))
    );
    let combined = result.combined.as_ref().unwrap();
    assert!(close(combined.actions[0].worst_case_regret.unwrap(), 1.5));
    assert!(close(combined.actions[1].worst_case_regret.unwrap(), 1.5));

    let partial = [
        evaluated("s1", None, &S1),
        evaluated("s4", None, &S4),
        atom("s5", None, AtomEvidence::Unevaluated("budget".into())),
    ];
    let result = run(0.0, 2.0, worst, PreferencePolicy::Maximin, &partial);
    assert_eq!(
        result.verdict,
        StructuralPreferenceVerdict::InsufficientScience(WORST_CASE_NEEDS_EVERY_STRUCTURE.into())
    );
    assert!(result.combined.is_some());
    assert_eq!(result.atoms[2].status, PreferenceAtomStatus::Unevaluated("budget".into()));
}

#[test]
fn weighting_over_structures_mixes_coefficients_without_renormalizing() {
    let bayes = StructuralPolicy::BayesOverStructures;
    let policy = PreferencePolicy::RequireInvariantBestAction;
    let result = run(
        0.0,
        1.0,
        bayes,
        policy,
        &[evaluated("s1", Some(0.5), &S1), evaluated("s3", Some(0.5), &S3)],
    );
    let combined = result.combined.as_ref().unwrap();
    let a = combined.actions[0].coefficients.as_ref().unwrap();
    assert!(close(a[0], 8.0) && close(a[1], -4.0), "{a:?}");
    assert_eq!(
        result.verdict,
        StructuralPreferenceVerdict::Combined(PreferenceVerdict::PreferenceDependent {
            possibly_best: both()
        })
    );
    assert_eq!(result.evaluated_mass, Some(1.0));
    assert_eq!(result.unidentified_mass, Some(0.0));
    assert_eq!(result.unevaluated_mass, Some(0.0));

    let result = run(
        0.0,
        1.0,
        bayes,
        policy,
        &[evaluated("s1", Some(0.5), &S1), evaluated("s2", Some(0.5), &S2)],
    );
    assert_eq!(
        result.verdict,
        StructuralPreferenceVerdict::Combined(PreferenceVerdict::NecessarilyBest("A".into()))
    );

    let unresolved =
        [evaluated("s1", Some(0.6), &S1), atom("s2", Some(0.4), AtomEvidence::Unidentified)];
    let result = run(0.0, 1.0, bayes, policy, &unresolved);
    assert_eq!(
        result.verdict,
        StructuralPreferenceVerdict::InsufficientScience(UNRESOLVED_MASS_CAN_CHANGE_BAYES.into())
    );
    assert!(result.unidentified_mass.is_some_and(|m| close(m, 0.4)));
    let a = result.combined.as_ref().unwrap().actions[0].coefficients.clone().unwrap();
    assert!(close(a[0], 6.0) && close(a[1], -2.4), "{a:?}");

    // Undeclared mass counts as unevaluated.
    let result = run(0.0, 1.0, bayes, policy, &[evaluated("s1", Some(0.5), &S1)]);
    assert!(result.unevaluated_mass.is_some_and(|m| close(m, 0.5)));
    assert!(matches!(result.verdict, StructuralPreferenceVerdict::InsufficientScience(_)));
}

#[test]
fn report_only_returns_each_structure_alone() {
    let result = run(
        0.0,
        1.0,
        StructuralPolicy::ReportOnly,
        PreferencePolicy::RequireInvariantBestAction,
        &[evaluated("s1", None, &S1), evaluated("s3", None, &S3)],
    );
    assert_eq!(result.verdict, StructuralPreferenceVerdict::ReportOnly);
    assert!(result.combined.is_none());
    assert_eq!(atom_verdict(&result, 0), PreferenceVerdict::NecessarilyBest("A".into()));
    assert_eq!(
        atom_verdict(&result, 1),
        PreferenceVerdict::PreferenceDependent { possibly_best: both() }
    );
}

#[test]
fn an_action_excluded_in_one_structure_is_excluded_in_all() {
    let mut c = contract_for(
        model(0.0, 1.0),
        2,
        StructuralPolicy::RequireInvariantBestAction,
        PreferencePolicy::RequireInvariantBestAction,
    );
    // A costs 4 in s1 and 2 in s2; B costs 1 in both.
    c.constraints = vec![HardConstraint {
        id: "budget".into(),
        expr: UtilityExpr::Input(1),
        bound: 3.0,
        min_probability: 1.0,
        units: "usd".into(),
        applies_to: vec![],
    }];
    let atoms = [evaluated("s1", None, &S1), evaluated("s2", None, &S2)];
    let result = evaluate_preference_structural(&c, &atoms, None, options()).unwrap();
    assert_eq!(
        result.verdict,
        StructuralPreferenceVerdict::Combined(PreferenceVerdict::NecessarilyBest("B".into()))
    );
    let PreferenceAtomStatus::Evaluated(second) = &result.atoms[1].status else {
        panic!("s2 was evaluated");
    };
    assert!(!second.actions[0].admissible);
    assert!(second.actions[0].exclusions.is_empty());
    let combined = result.combined.as_ref().unwrap();
    assert!(!combined.actions[0].admissible);
    assert_eq!(combined.actions[0].exclusions[0].constraint_id, "budget");
}

#[test]
fn evidence_that_cannot_answer_keeps_its_structure_unevaluated() {
    let m = model(0.0, 1.0);
    // Only A's coordinates: B's quantities are not found.
    let partial = law(&m, &[vec![vec![1.0, 1.0], vec![1.0, 1.0]]]);
    let atoms = [evaluated("s1", None, &S1), atom("s2", None, AtomEvidence::Evaluated(partial))];
    let result = run(
        0.0,
        1.0,
        StructuralPolicy::RequireInvariantBestAction,
        PreferencePolicy::RequireInvariantBestAction,
        &atoms,
    );
    assert!(matches!(result.atoms[1].status, PreferenceAtomStatus::Unevaluated(_)));
    assert_eq!(
        result.verdict,
        StructuralPreferenceVerdict::InsufficientScience(INVARIANCE_UNCHECKED.into())
    );
}

#[test]
fn every_structure_refusal_names_its_cause() {
    let c = |structural| {
        contract_for(model(0.0, 1.0), 2, structural, PreferencePolicy::RequireInvariantBestAction)
    };
    let invariant = c(StructuralPolicy::RequireInvariantBestAction);
    let refuse = |contract: &PreferenceContract, atoms: &[StructuralAtom]| {
        evaluate_preference_structural(contract, atoms, None, options()).unwrap_err()
    };
    let cases: Vec<(PreferenceContract, Vec<StructuralAtom>, E)> = vec![
        (invariant.clone(), vec![], E::NoStructures),
        (
            invariant.clone(),
            vec![evaluated("s1", None, &S1), evaluated(" ", None, &S2)],
            E::StructureIdBlank { position: 1 },
        ),
        (
            invariant.clone(),
            vec![evaluated("s1", None, &S1), evaluated("s1", None, &S2)],
            E::StructureDuplicate { id: "s1".into() },
        ),
        (
            invariant.clone(),
            vec![evaluated("s1", Some(0.5), &S1), evaluated("s2", None, &S2)],
            E::StructureProbabilitiesPartial,
        ),
        (
            invariant.clone(),
            vec![evaluated("s1", Some(-0.1), &S1), evaluated("s2", Some(0.5), &S2)],
            E::StructureProbabilityInvalid { structure: "s1".into() },
        ),
        (
            invariant,
            vec![evaluated("s1", Some(0.7), &S1), evaluated("s2", Some(0.6), &S2)],
            E::StructureProbabilitiesExceedOne,
        ),
        (
            c(StructuralPolicy::BayesOverStructures),
            vec![evaluated("s1", None, &S1), evaluated("s2", None, &S2)],
            E::StructureProbabilitiesRequired,
        ),
    ];
    let mut details = std::collections::BTreeSet::new();
    for (contract, atoms, expected) in cases {
        let refused = refuse(&contract, &atoms);
        assert_eq!(refused, expected);
        details.insert(refused.detail());
    }
    assert_eq!(details.len(), 7);
    assert!(details.iter().all(|d| d.starts_with("preference_decision.structure")));

    let mut regret = c(StructuralPolicy::Maximin);
    regret.criterion = DecisionCriterion::ExpectedRegret;
    let refused = refuse(&regret, &[evaluated("s1", None, &S1)]);
    assert_eq!(
        refused,
        E::CertifiedRouteRequired { reason: CertifiedRouteReason::WorstCaseOfExpectedRegret }
    );
}

// ---------------------------------------------------------------------------------------------
// Brute-force grid cross-check
// ---------------------------------------------------------------------------------------------

/// A fixed-seed generator (`SplitMix64`) so the random problems are reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(n).unwrap()).unwrap()
    }

    /// A multiple of 1/4 in `[low / 4, high / 4]`.
    fn quarter(&mut self, low: i32, high: i32) -> f64 {
        let span = usize::try_from(high - low + 1).unwrap();
        f64::from(low + i32::try_from(self.below(span)).unwrap()) / 4.0
    }

    /// A point of `[lower, upper]` on a fine grid.
    fn within(&mut self, lower: f64, upper: f64) -> f64 {
        lower + (upper - lower) * self.below(1001) as f64 / 1000.0
    }
}

/// `x0 + sum_j theta_j x_{j+1}` with draws per structure, action and attribute.
struct Problem {
    model: ValueModel,
    n_actions: usize,
    /// `draws[s][a][k]`: the draws of attribute `k` under action `a` in structure `s`.
    draws: Vec<Vec<Vec<Vec<f64>>>>,
    probabilities: Vec<f64>,
}

fn problem(rng: &mut Rng) -> Problem {
    let dimension = 1 + rng.below(3);
    let n_actions = 2 + rng.below(3);
    let n_structures = 1 + rng.below(3);
    let attributes: Vec<Attribute> =
        (0..=dimension).map(|k| Attribute { id: format!("x{k}"), units: "u".into() }).collect();
    let parameters: Vec<PreferenceParameter> = (0..dimension)
        .map(|j| {
            let lower = rng.quarter(-8, 4);
            let width = rng.quarter(4, 12);
            parameter(&format!("t{j}"), lower, lower + width)
        })
        .collect();
    let mut relations = Vec::new();
    if rng.below(2) == 0 {
        // A relation scaled to the ranges that keeps a fat part of the box: in coordinates
        // normalized to [-1, 1], `sum_j s_j z_j <= sum_j |s_j| / 4`.
        let mut terms = Vec::new();
        let mut bound = 0.0;
        let mut spread = 0.0;
        for (j, p) in parameters.iter().enumerate() {
            let s = [-1.0, -0.5, 0.5, 1.0][rng.below(4)];
            let half = 0.5 * (p.upper - p.lower);
            let centre = 0.5 * (p.upper + p.lower);
            terms.push((j, s / half));
            bound += s / half * centre;
            spread += f64::abs(s);
        }
        relations.push(LinearRelation {
            id: "cut".into(),
            terms,
            sense: RelationSense::AtMost,
            bound: bound + 0.25 * spread,
            origin: RelationOrigin::Declared,
        });
    }
    let mut value = P::Attribute(0);
    for j in 0..dimension {
        value = P::sum(value, P::product(P::Parameter(j), P::Attribute(j + 1)));
    }
    let model = ValueModel {
        attributes,
        preferences: PreferenceSet { parameters, relations },
        value,
        utility_units: "utility".into(),
    };
    let mut draws: Vec<Vec<Vec<Vec<f64>>>> = Vec::with_capacity(n_structures);
    for _ in 0..n_structures {
        let mut structure = Vec::with_capacity(n_actions);
        for _ in 0..n_actions {
            let mut arm = Vec::with_capacity(dimension + 1);
            for _ in 0..=dimension {
                arm.push((0..3).map(|_| rng.quarter(-20, 20)).collect::<Vec<f64>>());
            }
            structure.push(arm);
        }
        draws.push(structure);
    }
    let probabilities = match n_structures {
        1 => vec![1.0],
        2 => vec![0.25, 0.75],
        _ => vec![0.25, 0.25, 0.5],
    };
    Problem { model, n_actions, draws, probabilities }
}

impl Problem {
    /// `coefficients[s][a]`: the mean of each attribute, which is the part expectation.
    fn coefficients(&self) -> Vec<Vec<Vec<f64>>> {
        self.draws
            .iter()
            .map(|structure| {
                structure
                    .iter()
                    .map(|arm| {
                        arm.iter()
                            .map(|d| d.iter().sum::<f64>() / d.len() as f64)
                            .collect::<Vec<f64>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn atoms(&self) -> Vec<StructuralAtom> {
        self.draws
            .iter()
            .zip(&self.probabilities)
            .enumerate()
            .map(|(s, (arms, p))| {
                atom(&format!("s{s}"), Some(*p), AtomEvidence::Evaluated(law(&self.model, arms)))
            })
            .collect()
    }
}

fn random_measure(rng: &mut Rng, set: &PreferenceSet) -> PreferenceMeasure {
    let mut points = Vec::new();
    while points.len() < 3 {
        let point: Vec<f64> = set.parameters.iter().map(|p| rng.within(p.lower, p.upper)).collect();
        if set.contains(&point, 0.0) {
            points.push(point);
        }
    }
    PreferenceMeasure {
        points,
        weights: Some((0..3).map(|_| rng.quarter(1, 8)).collect()),
        labels: None,
        origin: MeasureOrigin::Declared { assumption_id: "panel".into() },
        unresolved: vec![],
    }
}

struct Grid {
    points: Vec<Vec<f64>>,
    spacing: Vec<f64>,
}

fn grid(set: &PreferenceSet) -> Grid {
    let n: usize = match set.dimension() {
        1 => 401,
        2 => 61,
        _ => 21,
    };
    let mut points: Vec<Vec<f64>> = vec![Vec::new()];
    let mut spacing = Vec::new();
    for p in &set.parameters {
        let step = (p.upper - p.lower) / (n - 1) as f64;
        spacing.push(step);
        let axis: Vec<f64> =
            (0..n).map(|i| if i + 1 == n { p.upper } else { p.lower + step * i as f64 }).collect();
        points = points
            .into_iter()
            .flat_map(|prefix| {
                axis.iter().map(move |v| {
                    let mut next = prefix.clone();
                    next.push(*v);
                    next
                })
            })
            .collect();
    }
    points.retain(|p| set.contains(p, 0.0));
    Grid { points, spacing }
}

fn affine(c: &[f64], theta: &[f64]) -> f64 {
    c[0] + c[1..].iter().zip(theta).map(|(x, t)| x * t).sum::<f64>()
}

/// The worst of an action's pieces.
fn score(pieces: &[Vec<f64>], theta: &[f64]) -> f64 {
    pieces.iter().map(|c| affine(c, theta)).fold(f64::INFINITY, f64::min)
}

/// Largest change of the score between neighbouring grid points.
fn step_change(pieces: &[Vec<f64>], spacing: &[f64]) -> f64 {
    pieces
        .iter()
        .map(|c| c[1..].iter().zip(spacing).map(|(x, h)| x.abs() * h).sum::<f64>())
        .fold(0.0, f64::max)
}

fn near(v: f64) -> f64 {
    1e-7 * (1.0 + v.abs())
}

/// `family[member][action]` lists the pieces whose minimum is the action's score in that
/// member; the table is the range over every member (one member except for invariance).
#[allow(clippy::too_many_lines, reason = "every quantity of one table checked in turn")]
fn check_table(
    label: &str,
    result: &PreferenceDecisionResult,
    family: &[Vec<Vec<Vec<f64>>>],
    grid: &Grid,
    set: &PreferenceSet,
) {
    let m = result.actions.len();
    let rows: Vec<Vec<f64>> = family
        .iter()
        .flat_map(|member| {
            grid.points
                .iter()
                .map(move |t| (0..m).map(|a| score(&member[a], t)).collect::<Vec<f64>>())
        })
        .collect();
    // Every admissible point lies within 4.5 grid steps of a grid point inside the set (the
    // set holds the box shrunk about its centre and a fat cut), so 8 steps bound the gap.
    let slack: Vec<f64> = (0..m)
        .map(|a| {
            8.0 * family
                .iter()
                .map(|member| step_change(&member[a], &grid.spacing))
                .fold(0.0, f64::max)
                + 1e-9
        })
        .collect();
    for a in 0..m {
        let outcome = &result.actions[a];
        let id = &outcome.id;
        let (low, high) = rows
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), r| (lo.min(r[a]), hi.max(r[a])));
        let (lo, hi) = outcome.value_range.unwrap();
        assert!(
            lo <= low + near(low) && low <= lo + slack[a] + near(lo),
            "{label} {id}: low {lo} vs grid {low}"
        );
        assert!(
            hi >= high - near(high) && high >= hi - slack[a] - near(hi),
            "{label} {id}: high {hi} vs grid {high}"
        );
        let rivals = || (0..m).filter(move |b| *b != a);
        let grid_regret = rows
            .iter()
            .map(|r| rivals().map(|b| r[b] - r[a]).fold(0.0, f64::max))
            .fold(0.0, f64::max);
        let rival_slack = rivals().map(|b| slack[b]).fold(0.0, f64::max);
        let regret = outcome.worst_case_regret.unwrap();
        assert!(
            regret >= grid_regret - near(grid_regret)
                && regret <= grid_regret + slack[a] + rival_slack + near(regret),
            "{label} {id}: regret {regret} vs grid {grid_regret}"
        );
        if rows.iter().any(|r| rivals().all(|b| r[a] > r[b] + 1e-7)) {
            assert_eq!(outcome.possibly_best, Some(true), "{label} {id}: best at a grid point");
        }
        if outcome.necessarily_best {
            assert!(
                rows.iter().all(|r| rivals().all(|b| r[a] - r[b] >= -1e-7)),
                "{label} {id}: necessarily best but beaten at a grid point"
            );
        }
        if outcome.uniquely_best {
            assert!(outcome.necessarily_best, "{label} {id}: uniquely best implies never beaten");
            assert!(outcome.dominated_by.is_empty(), "{label} {id}: uniquely best and dominated");
        }
        for rival in &outcome.dominated_by {
            let b = result.actions.iter().position(|o| &o.id == rival).unwrap();
            assert!(
                rows.iter().all(|r| r[b] - r[a] >= -1e-7),
                "{label} {id}: dominated by {rival} but ahead of it at a grid point"
            );
        }
    }
    if let PreferenceVerdict::NeverBeaten { action, ties_with } = &result.verdict {
        let a = result.actions.iter().position(|o| &o.id == action).unwrap();
        for rival in ties_with {
            let b = result.actions.iter().position(|o| &o.id == rival).unwrap();
            let low = rows.iter().map(|r| r[a] - r[b]).fold(f64::INFINITY, f64::min);
            assert!(
                low <= slack[a] + slack[b] + near(low),
                "{label}: {action} ties {rival} but leads it by {low} on the grid"
            );
        }
    }
    for a in 0..m {
        for b in (a + 1)..m {
            let (low, high) =
                rows.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), r| {
                    (lo.min(r[a] - r[b]), hi.max(r[a] - r[b]))
                });
            let switch = result
                .switches
                .iter()
                .find(|s| s.first == result.actions[a].id && s.second == result.actions[b].id);
            if low < -1e-6 && high > 1e-6 {
                assert!(switch.is_some(), "{label}: pair {a} {b} changes order on the grid");
            }
            let Some(switch) = switch else { continue };
            let (lo, hi) = switch.margin_range;
            let gap = slack[a] + slack[b];
            assert!(
                lo <= low + near(low) && low <= lo + gap + near(lo),
                "{label}: margin low {lo} vs {low}"
            );
            assert!(
                hi >= high - near(high) && high >= hi - gap - near(hi),
                "{label}: margin high {hi} vs {high}"
            );
            assert!(set.contains(&switch.favours_first_at, 1e-7), "{label}: witness outside");
            assert!(set.contains(&switch.favours_second_at, 1e-7), "{label}: witness outside");
            check_threshold(label, switch, set);
            if let [member] = family {
                let at = |t: &[f64]| score(&member[a], t) - score(&member[b], t);
                assert!(
                    (at(&switch.favours_first_at) - hi).abs() <= near(hi),
                    "{label}: high witness"
                );
                assert!(
                    (at(&switch.favours_second_at) - lo).abs() <= near(lo),
                    "{label}: low witness"
                );
            }
        }
    }
}

/// A threshold is reported exactly when the margin varies with one parameter alone, and the
/// margin vanishes there with the other parameters at their midpoints.
fn check_threshold(label: &str, switch: &PreferenceSwitch, set: &PreferenceSet) {
    let c = &switch.margin_coefficients;
    if c.is_empty() {
        assert_eq!(switch.threshold, None, "{label}: threshold without coefficients");
        return;
    }
    let varying: Vec<usize> = (0..set.dimension())
        .filter(|&j| c[j + 1].abs() * (set.parameters[j].upper - set.parameters[j].lower) > 1e-9)
        .collect();
    match (varying.as_slice(), &switch.threshold) {
        ([p], Some(threshold)) => {
            assert_eq!(threshold.parameter_id, set.parameters[*p].id, "{label}");
            assert_eq!(threshold.first_leads_above, c[p + 1] > 0.0, "{label}");
            let point: Vec<f64> = set
                .parameters
                .iter()
                .enumerate()
                .map(|(j, q)| if j == *p { threshold.value } else { 0.5 * (q.lower + q.upper) })
                .collect();
            let at = affine(c, &point);
            assert!(at.abs() <= near(c[0]), "{label}: margin {at} at the threshold");
        }
        ([_], None) => panic!("{label}: a one-parameter margin has no threshold"),
        (_, threshold) => assert_eq!(threshold, &None, "{label}: several parameters"),
    }
}

/// Shares and weighted values recomputed from the measure points.
fn check_measure(
    label: &str,
    result: &PreferenceDecisionResult,
    member: &[Vec<Vec<f64>>],
    measure: &PreferenceMeasure,
) {
    let m = result.actions.len();
    let weights = measure.weights.clone().unwrap();
    let total: f64 = weights.iter().sum();
    let mut share = vec![0.0; m];
    let mut value = vec![0.0; m];
    for (point, weight) in measure.points.iter().zip(&weights) {
        let w = weight / total;
        let scores: Vec<f64> = (0..m).map(|a| score(&member[a], point)).collect();
        let best = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let leaders: Vec<usize> = (0..m).filter(|a| scores[*a] >= best - 1e-9).collect();
        for &leader in &leaders {
            share[leader] += w / leaders.len() as f64;
        }
        for a in 0..m {
            value[a] += w * scores[a];
        }
    }
    for a in 0..m {
        let outcome = &result.actions[a];
        assert!(
            close(outcome.measure_share.unwrap(), share[a]),
            "{label}: share of {}",
            outcome.id
        );
        assert!(
            close(outcome.measure_value.unwrap(), value[a]),
            "{label}: value of {}",
            outcome.id
        );
    }
    assert_eq!(result.unresolved_measure_share, Some(0.0));
}

/// The verdict the policy rules give for the reported table.
fn expected_verdict(
    result: &PreferenceDecisionResult,
    policy: PreferencePolicy,
) -> PreferenceVerdict {
    let actions = &result.actions;
    let all: Vec<usize> = (0..actions.len()).collect();
    let pick = |score: &dyn Fn(usize) -> f64| -> Result<usize, Vec<usize>> {
        let best = all.iter().map(|a| score(*a)).fold(f64::NEG_INFINITY, f64::max);
        let tied: Vec<usize> = all.iter().copied().filter(|a| score(*a) >= best - 1e-9).collect();
        if tied.len() == 1 { Ok(tied[0]) } else { Err(tied) }
    };
    let ids = |list: Vec<usize>| -> Vec<String> {
        list.into_iter().map(|a| actions[a].id.clone()).collect()
    };
    match policy {
        PreferencePolicy::ReportOnly => PreferenceVerdict::ReportOnly,
        PreferencePolicy::Maximin => match pick(&|a| actions[a].value_range.unwrap().0) {
            Ok(a) => PreferenceVerdict::WorstCaseChoice(actions[a].id.clone()),
            Err(tied) => PreferenceVerdict::Tied(ids(tied)),
        },
        PreferencePolicy::MinimaxRegret => {
            match pick(&|a| -actions[a].worst_case_regret.unwrap()) {
                Ok(a) => PreferenceVerdict::MinimaxRegretChoice(actions[a].id.clone()),
                Err(tied) => PreferenceVerdict::Tied(ids(tied)),
            }
        }
        PreferencePolicy::Weighted => match pick(&|a| actions[a].measure_value.unwrap()) {
            Ok(a) => PreferenceVerdict::MeasureChoice {
                action: actions[a].id.clone(),
                share: actions[a].measure_share.unwrap(),
            },
            Err(tied) => PreferenceVerdict::Tied(ids(tied)),
        },
        PreferencePolicy::RequireInvariantBestAction => {
            if let Some(best) = actions.iter().find(|o| o.uniquely_best) {
                return PreferenceVerdict::NecessarilyBest(best.id.clone());
            }
            let unbeaten: Vec<usize> =
                all.iter().copied().filter(|a| actions[*a].necessarily_best).collect();
            match unbeaten.as_slice() {
                [] => PreferenceVerdict::PreferenceDependent {
                    possibly_best: actions
                        .iter()
                        .filter(|o| o.possibly_best != Some(false))
                        .map(|o| o.id.clone())
                        .collect(),
                },
                // The rivals it ties are checked against the grid in `check_table`.
                [a] => PreferenceVerdict::NeverBeaten {
                    action: actions[*a].id.clone(),
                    ties_with: match &result.verdict {
                        PreferenceVerdict::NeverBeaten { ties_with, .. } => ties_with.clone(),
                        _ => Vec::new(),
                    },
                },
                several => PreferenceVerdict::Tied(ids(several.to_vec())),
            }
        }
    }
}

fn check_coefficients(label: &str, result: &PreferenceDecisionResult, expected: &[Vec<f64>]) {
    for (outcome, want) in result.actions.iter().zip(expected) {
        let got = outcome.coefficients.as_ref().unwrap();
        assert!(got.iter().zip(want).all(|(g, w)| close(*g, *w)), "{label}: {got:?} vs {want:?}");
    }
}

/// Action ids in declaration order, read from the combined table.
fn combined_ids(result: &StructuralPreferenceResult) -> Option<Vec<String>> {
    result.combined.as_ref().map(|c| c.actions.iter().map(|o| o.id.clone()).collect())
}

/// The verdict of evaluated structures that agree, or each structure's verdict.
fn agreement(result: &StructuralPreferenceResult) -> StructuralPreferenceVerdict {
    let verdicts: Vec<AtomChoice> = (0..result.atoms.len())
        .map(|s| AtomChoice { atom: result.atoms[s].id.clone(), verdict: atom_verdict(result, s) })
        .collect();
    let first = verdicts[0].verdict.clone();
    if verdicts.iter().all(|v| v.verdict == first) {
        return StructuralPreferenceVerdict::Combined(first);
    }
    // One action unbeaten in every structure is never beaten across them.
    let unbeaten: Option<Vec<(&String, &[String])>> = verdicts
        .iter()
        .map(|v| match &v.verdict {
            PreferenceVerdict::NecessarilyBest(a) => Some((a, &[][..])),
            PreferenceVerdict::NeverBeaten { action, ties_with } => {
                Some((action, ties_with.as_slice()))
            }
            _ => None,
        })
        .collect();
    if let Some(list) = unbeaten
        && list.iter().all(|(a, _)| *a == list[0].0)
    {
        let ties_with = combined_ids(result)
            .unwrap_or_default()
            .into_iter()
            .filter(|id| list.iter().any(|(_, ties)| ties.contains(id)))
            .collect();
        return StructuralPreferenceVerdict::Combined(PreferenceVerdict::NeverBeaten {
            action: list[0].0.clone(),
            ties_with,
        });
    }
    if let PreferenceVerdict::MeasureChoice { action, .. } = &first {
        let shares: Option<Vec<f64>> = verdicts
            .iter()
            .map(|v| match &v.verdict {
                PreferenceVerdict::MeasureChoice { action: other, share } if other == action => {
                    Some(*share)
                }
                _ => None,
            })
            .collect();
        if let Some(shares) = shares {
            let share = shares.into_iter().fold(f64::INFINITY, f64::min);
            return StructuralPreferenceVerdict::Combined(PreferenceVerdict::MeasureChoice {
                action: action.clone(),
                share,
            });
        }
    }
    StructuralPreferenceVerdict::StructuresDisagree(verdicts)
}

#[test]
#[allow(clippy::too_many_lines, reason = "one loop over problems and every policy combination")]
fn every_linear_program_quantity_agrees_with_a_dense_grid() {
    let mut rng = Rng(0x5EED_C0FF_EE00_2026);
    let preferences = [
        PreferencePolicy::RequireInvariantBestAction,
        PreferencePolicy::Maximin,
        PreferencePolicy::MinimaxRegret,
        PreferencePolicy::Weighted,
        PreferencePolicy::ReportOnly,
    ];
    let structurals = [
        StructuralPolicy::ReportOnly,
        StructuralPolicy::BayesOverStructures,
        StructuralPolicy::Maximin,
        StructuralPolicy::RequireInvariantBestAction,
    ];
    for case in 0..10 {
        let problem = problem(&mut rng);
        let set = problem.model.preferences.clone();
        let grid = grid(&set);
        assert!(!grid.points.is_empty(), "case {case}: empty grid");
        let measure = random_measure(&mut rng, &set);
        let atoms = problem.atoms();
        let coefficients = problem.coefficients();
        let singles: Vec<Vec<Vec<Vec<f64>>>> = coefficients
            .iter()
            .map(|structure| structure.iter().map(|c| vec![c.clone()]).collect::<Vec<_>>())
            .collect();
        let mixture: Vec<Vec<f64>> = (0..problem.n_actions)
            .map(|a| {
                let mut mixed = vec![0.0; set.dimension() + 1];
                for (structure, p) in coefficients.iter().zip(&problem.probabilities) {
                    for (slot, c) in mixed.iter_mut().zip(&structure[a]) {
                        *slot += p * c;
                    }
                }
                mixed
            })
            .collect();
        let worst: Vec<Vec<Vec<f64>>> = (0..problem.n_actions)
            .map(|a| coefficients.iter().map(|structure| structure[a].clone()).collect())
            .collect();
        for structural in structurals {
            for preference in preferences {
                let label = format!("case {case} {structural:?} {preference:?}");
                let contract =
                    contract_for(problem.model.clone(), problem.n_actions, structural, preference);
                let result =
                    evaluate_preference_structural(&contract, &atoms, Some(&measure), options())
                        .unwrap();
                match structural {
                    StructuralPolicy::ReportOnly => {
                        assert_eq!(result.verdict, StructuralPreferenceVerdict::ReportOnly);
                        assert!(result.combined.is_none());
                        for (s, member) in singles.iter().enumerate() {
                            let PreferenceAtomStatus::Evaluated(single) = &result.atoms[s].status
                            else {
                                panic!("{label}: structure {s} was not evaluated");
                            };
                            let label = format!("{label} structure {s}");
                            check_table(&label, single, std::slice::from_ref(member), &grid, &set);
                            check_measure(&label, single, member, &measure);
                            check_coefficients(&label, single, &coefficients[s]);
                            assert_eq!(
                                single.verdict,
                                expected_verdict(single, preference),
                                "{label}"
                            );
                        }
                    }
                    StructuralPolicy::BayesOverStructures | StructuralPolicy::Maximin => {
                        let combined = result.combined.as_ref().unwrap();
                        let member: Vec<Vec<Vec<f64>>> = if structural == StructuralPolicy::Maximin
                        {
                            worst.clone()
                        } else {
                            check_coefficients(&label, combined, &mixture);
                            mixture.iter().map(|c| vec![c.clone()]).collect()
                        };
                        check_table(&label, combined, std::slice::from_ref(&member), &grid, &set);
                        check_measure(&label, combined, &member, &measure);
                        assert_eq!(
                            combined.verdict,
                            expected_verdict(combined, preference),
                            "{label}"
                        );
                        let expected = if preference == PreferencePolicy::ReportOnly {
                            PreferenceVerdict::ReportOnly
                        } else {
                            combined.verdict.clone()
                        };
                        assert_eq!(
                            result.verdict,
                            StructuralPreferenceVerdict::Combined(expected),
                            "{label}"
                        );
                    }
                    StructuralPolicy::RequireInvariantBestAction => {
                        let combined = result.combined.as_ref().unwrap();
                        check_table(&label, combined, &singles, &grid, &set);
                        let expected = if preference == PreferencePolicy::ReportOnly {
                            StructuralPreferenceVerdict::Combined(PreferenceVerdict::ReportOnly)
                        } else {
                            agreement(&result)
                        };
                        assert_eq!(result.verdict, expected, "{label}");
                    }
                }
                assert_eq!(result.evaluated_mass, Some(1.0), "{label}");
            }
        }
    }
}

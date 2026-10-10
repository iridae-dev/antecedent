//! X8 transported static path-specific counterfactual: hand-derived oracle,
//! enumeration over a finite exogenous space, refusals and the nonrecoverability witness.
//!
//! Structural model (covariate Z, treatment A, mediator M, outcome Y):
//!   `M = 0.5 Z + (2 - 0.5 Z) A + U_M`
//!   `Y = (3 + 0.75 Z) A + 4 M + 1.5 Z + U_Y`
//! with a1 = 2, a0 = 0.5 (so a1 - a0 = 1.5). By hand, per unit with covariate z:
//!   NDE(z) = (3 + 0.75 z) * 1.5            = 4.5 + 1.125 z
//!   NIE(z) = 4 * (2 - 0.5 z) * 1.5         = 12 - 3 z
//!   total  = NDE + NIE                     = 16.5 - 1.875 z
//! Target law {0: 0.25, 1: 0.75} (E z = 0.75); source law {0: 0.4, 1: 0.4, 2: 0.2}
//! (E z = 0.8).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_counterfactual::transported_gate::{PopulationRole, RegimeFactorKey};
use antecedent_counterfactual::transported_path_specific::{
    AdditiveLinearScm, Affine, CovariateLaw, DeclaredAssumptions, EdgeAssignment, LinearMechanism,
    PathSpecificQuery, SelectionDiagram, SelectionNode, TransportedPathSpecificInput,
    TransportedPathSpecificRefusal, TransportedPathSpecificResult,
    evaluate_transported_path_specific,
};

const TOL: f64 = 1e-12;

fn aff(constant: f64, slopes: &[(&str, f64)]) -> Affine {
    Affine {
        constant,
        covariate_slopes: slopes.iter().map(|(n, s)| ((*n).to_owned(), *s)).collect(),
    }
}

fn model() -> AdditiveLinearScm {
    let m = LinearMechanism {
        intercept: Affine::constant(0.0),
        parents: vec![("z".to_owned(), aff(0.5, &[])), ("a".to_owned(), aff(2.0, &[("z", -0.5)]))],
    };
    let y = LinearMechanism {
        intercept: Affine::constant(0.0),
        parents: vec![
            ("a".to_owned(), aff(3.0, &[("z", 0.75)])),
            ("m".to_owned(), aff(4.0, &[])),
            ("z".to_owned(), aff(1.5, &[])),
        ],
    };
    AdditiveLinearScm {
        treatment: "a".to_owned(),
        covariates: vec!["z".to_owned()],
        mechanisms: vec![("m".to_owned(), m), ("y".to_owned(), y)],
    }
}

fn law(points: &[(f64, f64)]) -> CovariateLaw {
    CovariateLaw {
        points: points.iter().map(|(z, w)| (BTreeMap::from([("z".to_owned(), *z)]), *w)).collect(),
    }
}

fn target() -> CovariateLaw {
    law(&[(0.0, 0.25), (1.0, 0.75)])
}

fn source() -> CovariateLaw {
    law(&[(0.0, 0.4), (1.0, 0.4), (2.0, 0.2)])
}

fn query(plus: EdgeAssignment, minus: EdgeAssignment) -> PathSpecificQuery {
    PathSpecificQuery {
        outcome: "y".to_owned(),
        treated_value: 2.0,
        control_value: 0.5,
        plus,
        minus,
    }
}

fn nde() -> PathSpecificQuery {
    query(EdgeAssignment::treated_on(["y"]), EdgeAssignment::all_control())
}

fn nie() -> PathSpecificQuery {
    query(EdgeAssignment::treated_on(["m", "y"]), EdgeAssignment::treated_on(["y"]))
}

fn total() -> PathSpecificQuery {
    query(EdgeAssignment::treated_on(["m", "y"]), EdgeAssignment::all_control())
}

fn declared() -> DeclaredAssumptions {
    DeclaredAssumptions {
        additive_noise: true,
        noise_laws_shared: true,
        cross_world_independence: true,
    }
}

fn factor(role: PopulationRole) -> RegimeFactorKey {
    RegimeFactorKey { role, regime: "observational".to_owned() }
}

fn evidence() -> BTreeMap<RegimeFactorKey, String> {
    [PopulationRole::Source, PopulationRole::Target]
        .into_iter()
        .map(|role| (factor(role), "evidence".to_owned()))
        .collect()
}

fn selection(label: &str, target: &str) -> SelectionDiagram {
    SelectionDiagram {
        selections: vec![SelectionNode { label: label.to_owned(), target: target.to_owned() }],
    }
}

type Outcome = Result<TransportedPathSpecificResult, Box<TransportedPathSpecificRefusal>>;

fn run_full(
    model: &AdditiveLinearScm,
    diagram: &SelectionDiagram,
    source: &CovariateLaw,
    target: &CovariateLaw,
    query: &PathSpecificQuery,
    assumptions: DeclaredAssumptions,
    factors: &BTreeMap<RegimeFactorKey, String>,
) -> Outcome {
    evaluate_transported_path_specific(&TransportedPathSpecificInput {
        model,
        diagram,
        source_law: source,
        target_law: target,
        query,
        assumptions,
        supplied_factors: factors,
    })
}

fn run(query: &PathSpecificQuery, target: &CovariateLaw) -> Outcome {
    run_full(&model(), &selection("S", "z"), &source(), target, query, declared(), &evidence())
}

fn refusal_of(outcome: Outcome) -> Box<TransportedPathSpecificRefusal> {
    outcome.expect_err("must refuse")
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < TOL, "{a} != {b}");
}

#[test]
fn x8t_hand_values_nde_nie_total() {
    let nde_result = run(&nde(), &target()).unwrap();
    close(nde_result.target_contrast, 4.5 + 1.125 * 0.75);
    close(nde_result.target_contrast, 5.34375);
    let nie_result = run(&nie(), &target()).unwrap();
    close(nie_result.target_contrast, 12.0 - 3.0 * 0.75);
    close(nie_result.target_contrast, 9.75);
    let total_result = run(&total(), &target()).unwrap();
    close(total_result.target_contrast, 15.09375);
    close(total_result.target_contrast, nde_result.target_contrast + nie_result.target_contrast);
    // Per-unit contrasts at the target support points.
    let by_z: Vec<f64> = nde_result.unit_contrasts.iter().map(|u| u.contrast).collect();
    close(by_z[0], 4.5);
    close(by_z[1], 5.625);
    assert_eq!(nde_result.derivation.claim, "point_only");
    assert!(nde_result.derivation.declared.contains(&"additive_noise"));
    assert!(nde_result.derivation.checked.contains(&"no_selection_on_mediator_or_outcome"));
}

/// Independent oracle: enumerate the exogenous space `(z, u_M, u_Y)` of the target
/// population with its own nested counterfactual code, not the evaluator's.
#[test]
fn x8t_enumeration_over_the_noise_space_matches_and_noise_cancels() {
    let m_fn = |z: f64, a: f64, u: f64| 0.5 * z + (2.0 - 0.5 * z) * a + u;
    let y_fn = |z: f64, a: f64, m: f64, u: f64| (3.0 + 0.75 * z) * a + 4.0 * m + 1.5 * z + u;
    let (a1, a0) = (2.0, 0.5);
    let (mut nde_sum, mut nie_sum) = (0.0, 0.0);
    for (z, pz) in [(0.0, 0.25), (1.0, 0.75)] {
        for (um, pm) in [(-1.0, 0.5), (1.0, 0.5)] {
            for (uy, py) in [(0.0, 0.5), (2.0, 0.5)] {
                let p = pz * pm * py;
                let (m1, m0) = (m_fn(z, a1, um), m_fn(z, a0, um));
                nde_sum += p * (y_fn(z, a1, m0, uy) - y_fn(z, a0, m0, uy));
                nie_sum += p * (y_fn(z, a1, m1, uy) - y_fn(z, a1, m0, uy));
            }
        }
    }
    close(run(&nde(), &target()).unwrap().target_contrast, nde_sum);
    close(run(&nie(), &target()).unwrap().target_contrast, nie_sum);
}

#[test]
fn x8t_target_law_change_moves_the_answer_by_exactly_the_hand_amount() {
    let base = run(&nde(), &target()).unwrap().target_contrast;
    // Move mass from z=1 to z=0: E z goes 0.75 -> 0.5, NDE slope is 1.125.
    let moved = run(&nde(), &law(&[(0.0, 0.5), (1.0, 0.5)])).unwrap().target_contrast;
    close(moved - base, 1.125 * (0.5 - 0.75));
    close(moved - base, -0.28125);
    // The same law as the source gives the source answer: the evaluator reads the target law.
    let as_source = run(&nde(), &source()).unwrap();
    close(as_source.target_contrast, 4.5 + 1.125 * 0.8);
    close(as_source.source_contrast, 5.4);
}

#[test]
fn x8t_source_only_answer_differs_from_the_target_answer() {
    let result = run(&nde(), &target()).unwrap();
    close(result.source_contrast, 5.4);
    close(result.target_contrast, 5.34375);
    assert!((result.source_contrast - result.target_contrast).abs() > 0.05);
}

#[test]
fn x8t_selection_on_covariate_or_treatment_is_allowed() {
    let diagram = SelectionDiagram {
        selections: vec![
            SelectionNode { label: "S1".to_owned(), target: "z".to_owned() },
            SelectionNode { label: "S2".to_owned(), target: "a".to_owned() },
        ],
    };
    let out = run_full(&model(), &diagram, &source(), &target(), &nde(), declared(), &evidence())
        .unwrap();
    close(out.target_contrast, 5.34375);
}

#[test]
fn x8t_selection_on_outcome_mechanism_refuses_with_a_two_model_witness() {
    let diagram = selection("S", "y");
    let refusal = refusal_of(run_full(
        &model(),
        &diagram,
        &source(),
        &target(),
        &nde(),
        declared(),
        &evidence(),
    ));
    assert_eq!(refusal.refusal.code, "transport_proven_non_transportable");
    assert_eq!(refusal.refusal.detail, "transported_counterfactual.selection_on_mechanism");
    assert!(refusal.refusal.validate().is_ok());
    let w = refusal.witness.expect("the NDE is sensitive to the outcome mechanism");
    assert_eq!(w.selected_node, "y");
    assert_eq!(w.perturbed_parent, "a");
    assert_eq!(w.perturbed_slope_covariate, None);
    // Both target models share the source model and agree on the source answer.
    assert_eq!(w.source_model, model());
    assert_eq!(w.target_model_a, w.source_model);
    close(w.source_contrast, 5.4);
    // Model A target: 5.34375. Model B raises the A coefficient of Y by 1, so the
    // NDE rises by 1 * (a1 - a0) = 1.5 under every covariate law.
    close(w.target_contrast_a, 5.34375);
    close(w.target_contrast_b, 5.34375 + 1.5);
    // Recompute model B's target answer independently, by hand: (3 + 1 + 0.75 z) * 1.5 averaged.
    let by_hand = (0.25 * 4.0 + 0.75 * 4.75) * 1.5;
    close(w.target_contrast_b, by_hand);
}

#[test]
fn x8t_selection_on_mediator_is_witnessed_only_when_the_contrast_depends_on_it() {
    let diagram = selection("S", "m");
    // The NDE does not depend on the mediator's equation: refused, no impossibility claimed.
    let refusal = refusal_of(run_full(
        &model(),
        &diagram,
        &source(),
        &target(),
        &nde(),
        declared(),
        &evidence(),
    ));
    assert_eq!(refusal.refusal.code, "cell_not_licensed");
    assert_eq!(refusal.refusal.detail, "transported_counterfactual.selection_on_mechanism");
    assert!(refusal.witness.is_none());
    assert_eq!(refusal.refusal.offending.as_deref(), Some("m"));
    // The NIE depends on it: witness. Perturbing b by 1 moves NIE by d * 1 * 1.5 = 6.
    let refusal = refusal_of(run_full(
        &model(),
        &diagram,
        &source(),
        &target(),
        &nie(),
        declared(),
        &evidence(),
    ));
    assert_eq!(refusal.refusal.code, "transport_proven_non_transportable");
    let w = refusal.witness.expect("witness");
    assert_eq!(w.selected_node, "m");
    assert_eq!(w.perturbed_parent, "a");
    close(w.target_contrast_a, 9.75);
    close(w.target_contrast_b, 15.75);
    close(w.source_contrast, 9.6);
}

#[test]
fn x8t_undeclared_premises_refuse_each_with_its_own_detail() {
    let go = |assumptions| {
        refusal_of(run_full(
            &model(),
            &selection("S", "z"),
            &source(),
            &target(),
            &nde(),
            assumptions,
            &evidence(),
        ))
    };
    let r = go(DeclaredAssumptions { additive_noise: false, ..declared() });
    assert_eq!(r.refusal.detail, "transported_counterfactual.nonadditive_mechanism");
    assert_eq!(r.refusal.code, "cell_not_licensed");
    let r = go(DeclaredAssumptions { noise_laws_shared: false, ..declared() });
    assert_eq!(r.refusal.detail, "transported_counterfactual.noise_law_not_shared");
    assert_eq!(r.refusal.offending.as_deref(), Some("noise_laws_shared"));
    assert!(r.refusal.validate().is_ok());
    let r = go(DeclaredAssumptions { cross_world_independence: false, ..declared() });
    assert_eq!(r.refusal.detail, "transported_counterfactual.cross_world_independence_missing");
    let r = go(DeclaredAssumptions::default());
    assert_eq!(r.refusal.detail, "transported_counterfactual.nonadditive_mechanism");
}

#[test]
fn x8t_target_outside_source_support_is_an_overlap_failure() {
    let outside = law(&[(0.0, 0.5), (3.0, 0.5)]);
    let r = refusal_of(run(&nde(), &outside));
    assert_eq!(r.refusal.code, "transport_support_failure");
    assert_eq!(r.refusal.detail, "transported_counterfactual.overlap_failure");
    assert_eq!(r.refusal.offending.as_deref(), Some("z=3"));
    assert!(r.refusal.validate().is_ok());
}

#[test]
fn x8t_absent_regime_factor_is_factor_missing() {
    let mut factors = evidence();
    factors.remove(&factor(PopulationRole::Target));
    let r = refusal_of(run_full(
        &model(),
        &selection("S", "z"),
        &source(),
        &target(),
        &nde(),
        declared(),
        &factors,
    ));
    assert_eq!(r.refusal.code, "transport_missing_evidence");
    assert_eq!(r.refusal.detail, "transported_counterfactual.factor_missing");
    assert_eq!(r.refusal.offending.as_deref(), Some("target:observational"));
    assert_eq!(r.missing_factors, vec![factor(PopulationRole::Target)]);
}

#[test]
fn x8t_malformed_inputs_are_invalid_arguments() {
    let mut cyclic = model();
    cyclic.mechanisms[0].1.parents.push(("y".to_owned(), aff(1.0, &[])));
    let r = refusal_of(run_full(
        &cyclic,
        &selection("S", "z"),
        &source(),
        &target(),
        &nde(),
        declared(),
        &evidence(),
    ));
    assert_eq!(r.refusal.code, "invalid_argument");
    assert_eq!(r.refusal.detail, "transported_counterfactual.invalid_model");
    let bad_query = query(EdgeAssignment::treated_on(["z"]), EdgeAssignment::all_control());
    let r = refusal_of(run(&bad_query, &target()));
    assert_eq!(r.refusal.detail, "transported_counterfactual.invalid_query");
    let r = refusal_of(run(&nde(), &law(&[(0.0, 0.5), (1.0, 0.25)])));
    assert_eq!(r.refusal.detail, "transported_counterfactual.invalid_law");
    let r = refusal_of(run_full(
        &model(),
        &selection("S", "nowhere"),
        &source(),
        &target(),
        &nde(),
        declared(),
        &evidence(),
    ));
    assert_eq!(r.refusal.detail, "transported_counterfactual.invalid_diagram");
}

#[test]
fn x8t_variable_order_does_not_change_the_answer() {
    let base = model();
    let mut shuffled = model();
    shuffled.mechanisms.reverse();
    for (_, mechanism) in &mut shuffled.mechanisms {
        mechanism.parents.reverse();
    }
    shuffled.covariates = vec!["w".to_owned(), "z".to_owned()];
    let with_w = |points: &[(f64, f64)]| CovariateLaw {
        points: points
            .iter()
            .map(|(z, p)| (BTreeMap::from([("z".to_owned(), *z), ("w".to_owned(), 7.0)]), *p))
            .collect(),
    };
    for q in [nde(), nie(), total()] {
        let expected = run_full(
            &base,
            &selection("S", "z"),
            &source(),
            &target(),
            &q,
            declared(),
            &evidence(),
        )
        .unwrap();
        let got = run_full(
            &shuffled,
            &selection("S", "z"),
            &with_w(&[(2.0, 0.2), (1.0, 0.4), (0.0, 0.4)]),
            &with_w(&[(1.0, 0.75), (0.0, 0.25)]),
            &q,
            declared(),
            &evidence(),
        )
        .unwrap();
        close(expected.target_contrast, got.target_contrast);
        close(expected.source_contrast, got.source_contrast);
    }
}

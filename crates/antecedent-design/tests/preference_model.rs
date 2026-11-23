//! Value models: evaluation, validation refusals, the exact linear form, shape parameters,
//! identity, preference-set geometry and piecewise-linear curves.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_design::decision_contract::UtilityExpr;
use antecedent_design::preference_model::{
    Attribute, Curve, LinearForm, LinearRelation, PreferenceExpr as P, PreferenceModelError as E,
    PreferenceParameter, PreferenceSet, RelationOrigin, RelationSense, ValueModel, shape,
};

fn attribute(id: &str, units: &str) -> Attribute {
    Attribute { id: id.into(), units: units.into() }
}

fn parameter(id: &str, lower: f64, upper: f64) -> PreferenceParameter {
    PreferenceParameter {
        id: id.into(),
        units: "usd/unit".into(),
        lower,
        upper,
        assumption_id: format!("{id}-judgment"),
    }
}

fn relation(
    id: &str,
    terms: Vec<(usize, f64)>,
    sense: RelationSense,
    bound: f64,
) -> LinearRelation {
    LinearRelation { id: id.into(), terms, sense, bound, origin: RelationOrigin::Declared }
}

/// `value = w0 * benefit - w1 * cost - 0.5 * max(cost - 10, 0)`, with `w0 >= 2 w1`.
fn model() -> ValueModel {
    ValueModel {
        attributes: vec![attribute("benefit", "events"), attribute("cost", "usd")],
        preferences: PreferenceSet {
            parameters: vec![parameter("w_benefit", 1.0, 3.0), parameter("w_cost", 0.5, 1.0)],
            relations: vec![
                relation("ratio", vec![(0, 1.0), (1, -2.0)], RelationSense::AtLeast, 0.0),
                relation("cap", vec![(0, 1.0)], RelationSense::AtMost, 2.5),
            ],
        },
        value: P::difference(
            P::difference(
                P::product(P::Parameter(0), shape::rate(0)),
                P::product(P::Parameter(1), shape::rate(1)),
            ),
            P::product(P::Const(0.5), shape::excess(1, 10.0)),
        ),
        utility_units: "usd".into(),
    }
}

/// Deterministic generator for property checks (splitmix64), seeded explicitly.
struct Draws(u64);

impl Draws {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self, low: f64, high: f64) -> f64 {
        let unit = (self.next() >> 11) as f64 / (1_u64 << 53) as f64;
        low + (high - low) * unit
    }

    fn below(&mut self, n: u64) -> usize {
        usize::try_from(self.next() % n).unwrap()
    }
}

fn close(a: f64, b: f64, scale: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + scale.abs())
}

#[test]
fn shapes_evaluate_to_their_definitions() {
    let x = [3.0, -1.5];
    let at = |e: &P| e.evaluate(&x, &[]).unwrap();
    assert!(close(at(&shape::rate(0)), 3.0, 0.0));
    assert!(close(at(&shape::segment(0, 1.0, 2.5)), 1.5, 0.0));
    assert!(close(at(&shape::segment(0, 4.0, 6.0)), 0.0, 0.0));
    assert!(close(at(&shape::segment(1, -2.0, 0.0)), 0.5, 0.0));
    assert!(close(at(&shape::excess(0, 2.0)), 1.0, 0.0));
    assert!(close(at(&shape::excess(1, 2.0)), 0.0, 0.0));
    assert!(close(at(&shape::shortfall(1, 0.5)), 2.0, 0.0));
    assert!(close(at(&shape::shortfall(0, 0.5)), 0.0, 0.0));
    assert!(close(at(&shape::interaction(0, 1)), -4.5, 0.0));
    assert!(close(at(&shape::both(0, 1)), -1.5, 0.0));
    let value = model().value_at(&[4.0, 12.0], &[2.0, 0.75]).unwrap();
    // 2 * 4 - 0.75 * 12 - 0.5 * 2
    assert!(close(value, -2.0, 0.0));
}

#[test]
fn evaluation_refuses_unknown_indices_and_non_finite_values() {
    assert_eq!(P::Attribute(2).evaluate(&[1.0], &[]), Err(E::UnknownAttribute { index: 2 }));
    assert_eq!(P::Parameter(1).evaluate(&[], &[1.0]), Err(E::UnknownParameter { index: 1 }));
    assert_eq!(P::ln(P::Const(0.0)).evaluate(&[], &[]), Err(E::NonFiniteValue));
    assert_eq!(P::ln(P::Attribute(0)).evaluate(&[-1.0], &[]), Err(E::NonFiniteValue));
    assert_eq!(P::exp(P::Const(1_000.0)).evaluate(&[], &[]), Err(E::NonFiniteValue));
    assert_eq!(
        P::product(P::Const(1e200), P::Const(1e200)).evaluate(&[], &[]),
        Err(E::NonFiniteValue)
    );
    assert_eq!(P::Attribute(0).evaluate(&[f64::NAN], &[]), Err(E::NonFiniteValue));
    assert!(close(P::ln(P::exp(P::Const(1.5))).evaluate(&[], &[]).unwrap(), 1.5, 0.0));
    let m = model();
    assert_eq!(m.value_at(&[1.0], &[1.0, 1.0]), Err(E::AttributeCount { expected: 2, found: 1 }));
    assert_eq!(m.value_at(&[1.0, 1.0], &[1.0]), Err(E::ParameterCount { expected: 2, found: 1 }));
}

#[test]
fn binding_substitutes_parameters_and_attributes() {
    let expr = P::sum(P::product(P::Parameter(0), P::Attribute(0)), P::Parameter(1));
    let bound = expr.bind_parameters(&[2.0, -1.0]).unwrap();
    assert_eq!(bound, P::sum(P::product(P::Const(2.0), P::Attribute(0)), P::Const(-1.0)));
    assert!(bound.parameters_used().is_empty());
    assert_eq!(expr.bind_parameters(&[2.0]), Err(E::UnknownParameter { index: 1 }));
    assert_eq!(
        expr.bind_parameters(&[2.0, f64::INFINITY]),
        Err(E::NonFiniteParameterValue { index: 1 })
    );
    let fixed = expr.bind_attribute(0, 4.0);
    assert!(fixed.attributes_used().is_empty());
    assert!(close(fixed.evaluate(&[], &[2.0, -1.0]).unwrap(), 7.0, 0.0));
    assert_eq!(expr.bind_attribute(3, 1.0), expr);
}

#[test]
fn used_indices_are_ascending_and_distinct() {
    let expr = P::sum(
        P::product(P::Parameter(2), P::Attribute(3)),
        P::sum(P::product(P::Parameter(0), P::Attribute(1)), P::Attribute(3)),
    );
    assert_eq!(expr.attributes_used(), vec![1, 3]);
    assert_eq!(expr.parameters_used(), vec![0, 2]);
}

#[test]
fn affinity_in_attributes_reads_parameters_as_constants() {
    assert!(P::product(P::Parameter(0), P::Attribute(0)).is_affine_in_attributes());
    assert!(P::product(P::exp(P::Parameter(0)), P::Attribute(0)).is_affine_in_attributes());
    assert!(P::maximum(P::Parameter(0), P::Const(1.0)).is_affine_in_attributes());
    assert!(!shape::excess(0, 1.0).is_affine_in_attributes());
    assert!(!shape::interaction(0, 1).is_affine_in_attributes());
    assert!(!P::exp(P::Attribute(0)).is_affine_in_attributes());
    assert!(P::negate(P::difference(P::Attribute(0), P::Attribute(1))).is_affine_in_attributes());
}

#[test]
fn lowering_maps_attributes_to_inputs_and_refuses_what_remains() {
    let expr = P::sum(shape::excess(0, 1.0), P::negate(P::Attribute(2)));
    let lowered = expr.to_utility_expr(&[Some(1), None, Some(0)]).unwrap();
    let attributes = [3.0, 99.0, 0.5];
    let inputs = [0.5, 3.0];
    assert!(close(
        lowered.evaluate(&inputs).unwrap(),
        expr.evaluate(&attributes, &[]).unwrap(),
        0.0
    ));
    assert_eq!(lowered.inputs_used(), vec![0, 1]);
    assert!(expr.to_utility_expr(&[Some(1), None, None]).is_none());
    assert!(expr.to_utility_expr(&[Some(1)]).is_none());
    assert!(P::Parameter(0).to_utility_expr(&[]).is_none());
    assert!(P::exp(P::Const(0.0)).to_utility_expr(&[]).is_none());
    assert!(P::ln(P::Attribute(0)).to_utility_expr(&[Some(0)]).is_none());
    assert_eq!(P::Const(2.0).to_utility_expr(&[]), Some(UtilityExpr::Const(2.0)));
}

#[test]
fn a_well_formed_model_validates() {
    let m = model();
    assert_eq!(m.validate(), Ok(()));
    assert_eq!(m.preferences.validate(), Ok(()));
    assert_eq!(m.preferences.dimension(), 2);
}

#[test]
#[allow(clippy::type_complexity, reason = "a table of edits and their refusals")]
#[allow(clippy::too_many_lines, reason = "one table listing every refusal")]
fn every_model_refusal_names_its_cause() {
    let cases: Vec<(Box<dyn Fn(&mut ValueModel)>, E)> = vec![
        (Box::new(|m| m.attributes.clear()), E::NoAttributes),
        (Box::new(|m| m.attributes[1].id = " ".into()), E::BlankAttributeId { position: 1 }),
        (
            Box::new(|m| m.attributes[1].id = "benefit".into()),
            E::DuplicateAttribute { id: "benefit".into() },
        ),
        (
            Box::new(|m| m.attributes[0].units = String::new()),
            E::BlankAttributeUnits { attribute: "benefit".into() },
        ),
        (
            Box::new(|m| m.preferences.parameters[0].id = String::new()),
            E::BlankParameterId { position: 0 },
        ),
        (
            Box::new(|m| m.preferences.parameters[1].id = "w_benefit".into()),
            E::DuplicateParameter { id: "w_benefit".into() },
        ),
        (
            Box::new(|m| m.preferences.parameters[1].units = " ".into()),
            E::BlankParameterUnits { parameter: "w_cost".into() },
        ),
        (
            Box::new(|m| m.preferences.parameters[1].upper = f64::INFINITY),
            E::NonFiniteRange { parameter: "w_cost".into() },
        ),
        (
            Box::new(|m| m.preferences.parameters[0].lower = f64::NAN),
            E::NonFiniteRange { parameter: "w_benefit".into() },
        ),
        (
            Box::new(|m| m.preferences.parameters[0].lower = 4.0),
            E::ReversedRange { parameter: "w_benefit".into() },
        ),
        (
            Box::new(|m| m.preferences.parameters[0].assumption_id = String::new()),
            E::BlankAssumption { parameter: "w_benefit".into() },
        ),
        (
            Box::new(|m| m.preferences.relations[1].id = String::new()),
            E::BlankRelationId { position: 1 },
        ),
        (
            Box::new(|m| m.preferences.relations[1].id = "ratio".into()),
            E::DuplicateRelation { id: "ratio".into() },
        ),
        (
            Box::new(|m| {
                m.preferences.relations[0].origin =
                    RelationOrigin::Choice { record_id: "r1".into(), rejected_option: " ".into() }
            }),
            E::BlankRelationOrigin { relation: "ratio".into() },
        ),
        (
            Box::new(|m| m.preferences.relations[0].terms.clear()),
            E::EmptyRelation { relation: "ratio".into() },
        ),
        (
            Box::new(|m| m.preferences.relations[0].terms[1].0 = 5),
            E::RelationUnknownParameter { relation: "ratio".into(), index: 5 },
        ),
        (
            Box::new(|m| m.preferences.relations[0].terms[1].0 = 0),
            E::RelationRepeatedParameter { relation: "ratio".into(), index: 0 },
        ),
        (
            Box::new(|m| m.preferences.relations[0].terms[0].1 = f64::NAN),
            E::RelationNonFiniteCoefficient { relation: "ratio".into() },
        ),
        (
            Box::new(|m| m.preferences.relations[1].bound = f64::NEG_INFINITY),
            E::RelationNonFiniteBound { relation: "cap".into() },
        ),
        (Box::new(|m| m.utility_units = " ".into()), E::BlankUtilityUnits),
        (Box::new(|m| m.value = P::sum(m.value.clone(), P::Const(f64::NAN))), E::NonFiniteConstant),
        (
            Box::new(|m| {
                let mut deep = m.value.clone();
                for _ in 0..70 {
                    deep = P::negate(deep);
                }
                m.value = deep;
            }),
            E::ExpressionTooDeep { depth: 75 },
        ),
        (
            Box::new(|m| m.value = P::sum(m.value.clone(), P::Attribute(2))),
            E::UnknownAttribute { index: 2 },
        ),
        (
            Box::new(|m| m.value = P::sum(m.value.clone(), P::Parameter(4))),
            E::UnknownParameter { index: 4 },
        ),
        (
            Box::new(|m| m.preferences.parameters.push(parameter("w_spare", 0.0, 1.0))),
            E::UnusedParameter { parameter: "w_spare".into() },
        ),
        (
            Box::new(|m| m.attributes.push(attribute("harm", "events"))),
            E::UnusedAttribute { attribute: "harm".into() },
        ),
    ];
    let mut details = std::collections::BTreeSet::new();
    for (edit, expected) in cases {
        let mut m = model();
        edit(&mut m);
        let refused = m.validate().unwrap_err();
        assert_eq!(refused, expected);
        assert!(refused.detail().starts_with("preference_model."));
        assert!(m.identity().is_err());
        details.insert(refused.detail());
    }
    // One detail per distinct cause (the two range cases share one).
    assert_eq!(details.len(), 25);
}

#[test]
fn the_linear_form_is_an_exact_symbolic_decomposition() {
    let m = model();
    let LinearForm { fixed, terms } = m.linear_form().unwrap();
    assert_eq!(fixed, P::negate(P::product(P::Const(0.5), shape::excess(1, 10.0))));
    assert_eq!(terms, vec![P::Attribute(0), P::negate(P::Attribute(1))]);
    assert!(m.shape_parameters().is_empty());

    // A constant scale on a parameter and on an attribute, and an absent term.
    let scaled = ValueModel {
        value: P::sum(
            P::product(P::Const(3.0), P::Parameter(0)),
            P::product(P::sum(P::Parameter(0), P::Const(1.0)), P::Attribute(0)),
        ),
        attributes: vec![attribute("benefit", "events")],
        preferences: PreferenceSet {
            parameters: vec![parameter("w", 0.0, 1.0), parameter("unused", 0.0, 1.0)],
            relations: vec![],
        },
        utility_units: "usd".into(),
    };
    let form = scaled.linear_form().unwrap();
    assert_eq!(form.fixed, P::Attribute(0));
    assert_eq!(form.terms[0], P::sum(P::Const(3.0), P::Attribute(0)));
    assert_eq!(form.terms[1], P::Const(0.0));
}

#[test]
fn parameters_inside_kinks_or_products_of_parameters_block_the_linear_form() {
    let base = model();
    let cases = [
        (P::product(P::Parameter(0), P::product(P::Parameter(1), P::Attribute(0))), vec![0, 1]),
        (P::sum(P::maximum(P::Parameter(0), P::Attribute(0)), P::Parameter(1)), vec![0]),
        (P::product(P::exp(P::Parameter(1)), P::sum(P::Attribute(0), P::Parameter(0))), vec![0, 1]),
        (
            P::sum(
                P::product(P::Parameter(0), P::Attribute(0)),
                P::ln(P::sum(P::Parameter(1), P::Attribute(1))),
            ),
            vec![1],
        ),
        (P::minimum(P::Parameter(0), P::Parameter(1)), vec![0, 1]),
    ];
    for (value, shape_parameters) in cases {
        let m = ValueModel { value, ..base.clone() };
        assert!(m.linear_form().is_none());
        assert_eq!(m.shape_parameters(), shape_parameters);
    }
}

/// A random expression affine in the parameters, magnitudes kept small by bounded depth.
fn affine_tree(draws: &mut Draws, depth: usize, attributes: usize, parameters: usize) -> P {
    if depth == 0 {
        return match draws.below(2) {
            0 => P::Parameter(draws.below(parameters as u64)),
            _ => free_tree(draws, 1, attributes),
        };
    }
    let next = depth - 1;
    match draws.below(6) {
        0 => P::sum(
            affine_tree(draws, next, attributes, parameters),
            affine_tree(draws, next, attributes, parameters),
        ),
        1 => P::difference(
            affine_tree(draws, next, attributes, parameters),
            affine_tree(draws, next, attributes, parameters),
        ),
        2 => P::negate(affine_tree(draws, next, attributes, parameters)),
        3 => P::product(
            free_tree(draws, next, attributes),
            affine_tree(draws, next, attributes, parameters),
        ),
        4 => P::product(
            affine_tree(draws, next, attributes, parameters),
            free_tree(draws, next, attributes),
        ),
        _ => P::Parameter(draws.below(parameters as u64)),
    }
}

fn free_tree(draws: &mut Draws, depth: usize, attributes: usize) -> P {
    if depth == 0 {
        return match draws.below(2) {
            0 => P::Const(draws.uniform(-1.5, 1.5)),
            _ => P::Attribute(draws.below(attributes as u64)),
        };
    }
    let next = depth - 1;
    let (a, b) = (free_tree(draws, next, attributes), free_tree(draws, next, attributes));
    match draws.below(7) {
        0 => P::sum(a, b),
        1 => P::difference(a, b),
        2 => P::product(a, b),
        3 => P::minimum(a, b),
        4 => P::maximum(a, b),
        5 => P::exp(P::product(P::Const(0.25), a)),
        _ => P::negate(a),
    }
}

#[test]
fn the_linear_form_reproduces_the_value_at_random_points() {
    let mut draws = Draws(0x5EED_0001);
    let (attributes, parameters) = (3, 4);
    let base = ValueModel {
        attributes: (0..attributes).map(|k| attribute(&format!("a{k}"), "units")).collect(),
        preferences: PreferenceSet {
            parameters: (0..parameters).map(|j| parameter(&format!("p{j}"), -2.0, 2.0)).collect(),
            relations: vec![],
        },
        value: P::Const(0.0),
        utility_units: "usd".into(),
    };
    for _ in 0..300 {
        let value = affine_tree(&mut draws, 3, attributes, parameters);
        let m = ValueModel { value, ..base.clone() };
        let form = m.linear_form().expect("affine by construction");
        assert!(m.shape_parameters().is_empty());
        assert_eq!(form.terms.len(), parameters);
        for term in &form.terms {
            assert!(term.parameters_used().is_empty());
        }
        assert!(form.fixed.parameters_used().is_empty());
        for _ in 0..5 {
            let x: Vec<f64> = (0..attributes).map(|_| draws.uniform(-2.0, 2.0)).collect();
            let theta: Vec<f64> = (0..parameters).map(|_| draws.uniform(-2.0, 2.0)).collect();
            let want = m.value_at(&x, &theta).unwrap();
            let fixed = form.fixed.evaluate(&x, &[]).unwrap();
            let mut got = fixed;
            let mut scale = fixed.abs() + want.abs();
            for (term, weight) in form.terms.iter().zip(&theta) {
                let part = weight * term.evaluate(&x, &[]).unwrap();
                got += part;
                scale += part.abs();
            }
            assert!(close(got, want, scale), "{got} vs {want} for {:?}", m.value);
        }
    }
}

#[test]
fn identity_ignores_relation_and_term_order_but_not_semantics() {
    let base = model();
    let id = base.identity().unwrap();
    assert_eq!(id.len(), 64);
    let mut reordered = base.clone();
    reordered.preferences.relations.reverse();
    reordered.preferences.relations[1].terms.reverse();
    assert_eq!(reordered.identity().unwrap(), id);

    #[allow(clippy::type_complexity, reason = "a list of semantic edits")]
    let edits: Vec<Box<dyn Fn(&mut ValueModel)>> = vec![
        Box::new(|m| m.attributes[0].id = "benefit2".into()),
        Box::new(|m| m.attributes[1].units = "eur".into()),
        Box::new(|m| m.preferences.parameters[0].id = "w".into()),
        Box::new(|m| m.preferences.parameters[0].units = "eur/event".into()),
        Box::new(|m| m.preferences.parameters[0].lower = 1.5),
        Box::new(|m| m.preferences.parameters[1].upper = 2.0),
        Box::new(|m| m.preferences.parameters[1].assumption_id = "other".into()),
        Box::new(|m| m.preferences.relations[0].bound = 0.5),
        Box::new(|m| m.preferences.relations[0].terms[1].1 = -3.0),
        Box::new(|m| m.preferences.relations[1].sense = RelationSense::Equal),
        Box::new(|m| m.preferences.relations[1].id = "cap2".into()),
        Box::new(|m| m.preferences.relations[1].origin = RelationOrigin::ShapeRule),
        Box::new(|m| {
            m.preferences.relations[1].origin =
                RelationOrigin::Choice { record_id: "r".into(), rejected_option: "o".into() }
        }),
        Box::new(|m| {
            m.preferences.relations.pop();
        }),
        Box::new(|m| m.value = P::sum(m.value.clone(), P::Const(1.0))),
        Box::new(|m| m.utility_units = "eur".into()),
    ];
    let mut seen = std::collections::BTreeSet::from([id]);
    for edit in edits {
        let mut changed = base.clone();
        edit(&mut changed);
        assert!(seen.insert(changed.identity().unwrap()));
    }
}

#[test]
fn the_preference_set_is_a_box_cut_by_relations() {
    let set = model().preferences;
    assert_eq!(set.bounds(), vec![(1.0, 3.0), (0.5, 1.0)]);
    assert_eq!(set.midpoint(), vec![2.0, 0.75]);
    assert!(set.contains(&[2.0, 0.75], 0.0));
    assert!(!set.contains(&[2.75, 0.75], 0.0)); // above the cap
    assert!(set.contains(&[2.5 + 1e-10, 0.75], 1e-9));
    assert!(!set.contains(&[1.2, 0.75], 0.0)); // ratio violated
    assert!(!set.contains(&[2.0, 0.4], 0.0)); // below the box
    assert!(!set.contains(&[2.0], 0.0));
    assert!(!set.contains(&[2.0, f64::NAN], 1.0));
    assert!(!set.contains(&[2.0, 0.75], -1.0));

    let mut with_equal = set.clone();
    with_equal.relations.push(relation("tie", vec![(1, 2.0)], RelationSense::Equal, 1.5));
    let rows = with_equal.inequality_rows();
    assert_eq!(
        rows.coefficients,
        vec![vec![-1.0, 2.0], vec![1.0, 0.0], vec![0.0, 2.0], vec![0.0, -2.0]]
    );
    assert_eq!(rows.rhs, vec![-0.0, 2.5, 1.5, -1.5]);
    assert_eq!(rows.row_relation, vec![0, 1, 2, 2]);
    assert!(with_equal.contains(&[2.0, 0.75], 0.0));
    assert!(!with_equal.contains(&[2.0, 0.8], 0.0));
}

#[test]
fn degenerate_models_are_accepted() {
    // No parameters: the value is fixed and the linear form is its fixed part alone.
    let fixed = ValueModel {
        attributes: vec![attribute("benefit", "events")],
        preferences: PreferenceSet { parameters: vec![], relations: vec![] },
        value: shape::excess(0, 1.0),
        utility_units: "usd".into(),
    };
    assert_eq!(fixed.validate(), Ok(()));
    let form = fixed.linear_form().unwrap();
    assert!(form.terms.is_empty());
    assert_eq!(form.fixed, shape::excess(0, 1.0));
    assert!(fixed.preferences.contains(&[], 0.0));
    assert!(fixed.preferences.inequality_rows().coefficients.is_empty());
    // A fixed parameter (lower == upper) is valid.
    let mut point = model();
    point.preferences.parameters[1].lower = 0.75;
    point.preferences.parameters[1].upper = 0.75;
    assert_eq!(point.validate(), Ok(()));
    assert!(close(point.preferences.midpoint()[1], 0.75, 0.0));
}

#[test]
fn curves_are_piecewise_linear_with_shape_rules() {
    let curve =
        Curve { attribute: 0, breakpoints: vec![0.0, 1.0, 2.5, 4.0], slopes: vec![0, 1, 2] };
    assert_eq!(curve.validate(), Ok(()));
    let expr = curve.expr();
    let theta = [2.0, 1.0, 0.5];
    let at = |x: f64| expr.evaluate(&[x], &theta).unwrap();
    assert!(close(at(-1.0), 0.0, 0.0));
    assert!(close(at(0.5), 1.0, 0.0));
    assert!(close(at(2.0), 2.0 + 1.0, 0.0));
    assert!(close(at(3.0), 2.0 + 1.5 + 0.25, 0.0));
    assert!(close(at(9.0), 2.0 + 1.5 + 0.75, 0.0));

    let increasing = curve.increasing("benefit");
    assert_eq!(increasing.len(), 3);
    assert_eq!(increasing[2].id, "benefit.increasing.2");
    assert_eq!(increasing[2].terms, vec![(2, 1.0)]);
    assert_eq!(increasing[2].sense, RelationSense::AtLeast);
    assert_eq!(increasing[2].origin, RelationOrigin::ShapeRule);
    let decreasing = curve.decreasing("benefit");
    assert_eq!(decreasing[0].sense, RelationSense::AtMost);
    let diminishing = curve.diminishing("benefit");
    assert_eq!(diminishing.len(), 2);
    assert_eq!(diminishing[1].terms, vec![(1, 1.0), (2, -1.0)]);
    assert_eq!(diminishing[1].sense, RelationSense::AtLeast);
    let accelerating = curve.accelerating("benefit");
    assert_eq!(accelerating[0].terms, vec![(0, -1.0), (1, 1.0)]);

    // Every generated relation is valid in a set of the curve's parameters, and they hold
    // at diminishing increasing slopes.
    let mut relations = increasing;
    relations.extend(diminishing);
    let set = PreferenceSet {
        parameters: (0..3).map(|j| parameter(&format!("s{j}"), 0.0, 4.0)).collect(),
        relations,
    };
    assert_eq!(set.validate(), Ok(()));
    assert!(set.contains(&theta, 0.0));
    assert!(!set.contains(&[1.0, 2.0, 0.5], 0.0));

    // One slope shared by two segments gives no ordering relation between them.
    let shared = Curve { attribute: 0, breakpoints: vec![0.0, 1.0, 2.0], slopes: vec![0, 0] };
    assert!(shared.diminishing("s").is_empty());
    let curved = ValueModel {
        attributes: vec![attribute("benefit", "events")],
        preferences: PreferenceSet {
            parameters: (0..3).map(|j| parameter(&format!("s{j}"), 0.0, 4.0)).collect(),
            relations: vec![],
        },
        value: curve.expr(),
        utility_units: "usd".into(),
    };
    assert_eq!(curved.validate(), Ok(()));
    let form = curved.linear_form().unwrap();
    assert_eq!(form.fixed, P::Const(0.0));
    assert_eq!(form.terms[1], shape::segment(0, 1.0, 2.5));
}

#[test]
fn malformed_curves_refuse() {
    let curve =
        |breakpoints: Vec<f64>, slopes: Vec<usize>| Curve { attribute: 0, breakpoints, slopes };
    assert_eq!(curve(vec![0.0], vec![]).validate(), Err(E::CurveTooShort { breakpoints: 1 }));
    assert_eq!(
        curve(vec![0.0, 1.0], vec![0, 1]).validate(),
        Err(E::CurveSlopeCount { breakpoints: 2, slopes: 2 })
    );
    assert_eq!(
        curve(vec![0.0, f64::NAN], vec![0]).validate(),
        Err(E::CurveBreakpointNonFinite { position: 1 })
    );
    assert_eq!(
        curve(vec![0.0, 2.0, 2.0], vec![0, 1]).validate(),
        Err(E::CurveBreakpointsNotIncreasing { position: 2 })
    );
    for refused in [
        E::CurveTooShort { breakpoints: 1 },
        E::CurveSlopeCount { breakpoints: 2, slopes: 2 },
        E::CurveBreakpointNonFinite { position: 1 },
        E::CurveBreakpointsNotIncreasing { position: 2 },
        E::NonFiniteValue,
        E::AttributeCount { expected: 1, found: 2 },
        E::ParameterCount { expected: 1, found: 2 },
        E::NonFiniteParameterValue { index: 0 },
    ] {
        assert!(refused.detail().starts_with("preference_model."));
        assert!(!refused.to_string().is_empty());
    }
}

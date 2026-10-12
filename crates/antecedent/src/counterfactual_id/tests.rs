//! Tests of the counterfactual identification cell.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CounterfactualEventQuery, ExecutionContext, RegimeId, Value, VariableId};
use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, LawTolerance};
use antecedent_identify::counterfactual_id::{
    COUNTERFACTUAL_ID_DEFAULT_LIMITS, COUNTERFACTUAL_ID_MEMORY_BYTES, CounterfactualIdProblem,
    decide_counterfactual_id, evaluate_counterfactual_functional,
};

use super::artifact::{agrees, direct_sum_numerators, law_wire};

fn frontdoor_law(probabilities: Vec<f64>) -> ExactDiscreteLaw {
    let axes: Vec<DiscreteAxis> = (0..3)
        .map(|i| DiscreteAxis {
            variable: VariableId::from_raw(i),
            values: vec![Value::f64(0.0), Value::f64(1.0)].into(),
        })
        .collect();
    ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(0),
        [],
        axes,
        probabilities,
        "unit",
        LawTolerance::default(),
    )
    .unwrap()
}

/// The consumer's direct-sum evaluator reads masses straight from the law's
/// cells and walks the expressions with its own arithmetic: it reproduces
/// the compiled evaluation and the front-door formula, and on a law where a
/// conditional with positive weight is undefined it declines, as the
/// compiled evaluation refuses.
#[test]
fn the_direct_sum_evaluation_agrees_with_the_compiled_one_and_the_formula() {
    let ctx = ExecutionContext::for_tests(1);
    let problem =
        CounterfactualIdProblem::new(vec![vec![0.0, 1.0]; 3], &[(0, 1), (1, 2)], &[(0, 2)])
            .unwrap();
    let query = CounterfactualEventQuery::effect_on_treated(
        VariableId::from_raw(0),
        1.0,
        0.0,
        VariableId::from_raw(2),
        1.0,
    )
    .unwrap();
    let derivation = decide_counterfactual_id(
        &problem,
        &query,
        COUNTERFACTUAL_ID_DEFAULT_LIMITS,
        COUNTERFACTUAL_ID_MEMORY_BYTES,
        &ctx,
    )
    .unwrap();
    let levels = vec![vec![0u64, 1.0f64.to_bits()]; 3];
    let p = vec![0.10, 0.05, 0.08, 0.17, 0.12, 0.13, 0.06, 0.29];
    let law = frontdoor_law(p.clone());
    let point = evaluate_counterfactual_functional(&problem, &derivation, &law, &ctx).unwrap();
    let direct = direct_sum_numerators(&derivation, &law_wire(&law).unwrap(), &levels).unwrap();
    assert_eq!(direct.len(), point.numerators.len());
    for (d, (_, c)) in direct.iter().zip(&point.numerators) {
        assert!(agrees(*d, *c), "{d} vs {c}");
    }
    // P(Y_x = 1, X = 0) = P(x' = 0) sum_m P(m | x = 1) P(y = 1 | m, x' = 0).
    let cell = |x: usize, m: usize, y: usize| p[x * 4 + m * 2 + y];
    let px = |x: usize| (0..4).map(|k| p[x * 4 + k]).sum::<f64>();
    let pxm = |x: usize, m: usize| cell(x, m, 0) + cell(x, m, 1);
    let formula =
        px(0) * (0..2).map(|m| pxm(1, m) / px(1) * cell(0, m, 1) / pxm(0, m)).sum::<f64>();
    assert!(agrees(direct[1], formula), "{} vs {formula}", direct[1]);
    // P(X = 0, M = 1) = 0 while P(M = 1 | X = 1) > 0: both evaluators decline.
    let gap = frontdoor_law(vec![0.2, 0.2, 0.0, 0.0, 0.1, 0.1, 0.2, 0.2]);
    assert!(direct_sum_numerators(&derivation, &law_wire(&gap).unwrap(), &levels).is_none());
    assert!(evaluate_counterfactual_functional(&problem, &derivation, &gap, &ctx).is_err());
}

use antecedent_core::{
    AverageEffectQuery, CausalQuery, ContinuousDomain, GridSpec, Intervention, MechanismOverride,
    ResponseFunctional, ResponseQuery, TargetPopulation, Value, VariableId,
};
use antecedent_graph::{Admg, Dag, DenseNodeId};
use std::sync::Arc;

use super::*;
use crate::error::IdentificationError;
use crate::identifier::IdentificationWorkspace;
use crate::oracle_dot::{admg_from_oracle_dot, dag_from_oracle_dot};
use crate::result::IdentificationStatus;

fn chain_dag() -> Dag {
    let mut dag = Dag::with_variables(3);
    // 0 -> 1 -> 2  (T -> M -> Y) but use T=0, Z=1, Y=2 with Z confounder style:
    // backdoor chain: Z -> T -> Y, Z -> Y  => nodes 0=Z, 1=T, 2=Y
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    dag
}

#[test]
fn backdoor_chain_identified() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../conformance/identify/general_id_backdoor_chain/expected.json"
    ))
    .unwrap();
    assert_eq!(fixture["case"], "identifiable_backdoor");
    assert_eq!(fixture["treatment"], "t");
    assert_eq!(fixture["outcome"], "y");
    assert_eq!(fixture["expected_status_family"], "identified");
    assert!(
        fixture["reference"]["outputs"]["estimand"]
            .as_str()
            .unwrap()
            .contains("Estimand name: backdoor\nEstimand expression:"),
        "external oracle must identify the frozen Z -> T, Z -> Y, T -> Y graph by backdoor"
    );

    let (dag, nodes) = dag_from_oracle_dot(fixture["graph_dot"].as_str().unwrap());
    let id = IdIdentifier::new();
    let prep = id.prepare_dag(&dag).unwrap();
    let q = AverageEffectQuery::binary_ate(
        nodes.id(fixture["treatment"].as_str().unwrap()),
        nodes.id(fixture["outcome"].as_str().unwrap()),
    );
    let mut ws = IdentificationWorkspace::default();
    let res = id.identify_ate(&prep, &q, &mut ws).unwrap();
    assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
    assert_eq!(res.estimands[0].method_kind().unwrap(), EstimandMethod::GeneralId);
}

fn identify_chain_under(
    ctx: &antecedent_core::ExecutionContext,
) -> Result<(), IdentificationError> {
    let id = IdIdentifier::new();
    let prep = id.prepare_dag(&chain_dag()).unwrap();
    let q = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
    let mut ws = IdentificationWorkspace::default();
    antecedent_core::numerical_admission::with_context(ctx, || {
        id.identify_ate(&prep, &q, &mut ws).map(|_| ())
    })
}

#[test]
fn id_recursion_refuses_one_past_the_work_cap() {
    let base = antecedent_core::ExecutionContext::for_tests(1);
    // The smallest cap the identification fits under; one below it must refuse.
    let fits = (1..=4096u64)
        .find(|&cap| identify_chain_under(&base.with_id_work_limit(cap)).is_ok())
        .expect("the chain identifies under a small cap");
    assert!(fits > 1, "identification needs more than one recursive call");
    let past = identify_chain_under(&base.with_id_work_limit(fits - 1)).unwrap_err();
    assert!(matches!(
        past,
        IdentificationError::Budget { budget: crate::error::IdentificationBudget::Steps }
    ));
    assert!(identify_chain_under(&base.with_id_work_limit(0)).is_err());
}

#[test]
fn id_work_cap_is_finite_and_a_context_only_lowers_it() {
    let base = antecedent_core::ExecutionContext::for_tests(1);
    let default = antecedent_core::ExecutionContext::DEFAULT_ID_WORK_LIMIT;
    assert_eq!(base.id_work_limit(), default);
    assert_eq!(base.with_id_work_limit(u64::MAX).id_work_limit(), default);
    assert_eq!(base.with_id_work_limit(7).id_work_limit(), 7);
    identify_chain_under(&base).unwrap();
}

#[test]
fn id_recursion_polls_cancellation() {
    let ctx = antecedent_core::ExecutionContext::for_tests(1);
    ctx.cancellation.cancel();
    assert_eq!(identify_chain_under(&ctx).unwrap_err(), IdentificationError::Cancelled);
}

#[test]
fn general_id_records_causal_markov() {
    // Truncated-factorization / g-formula identification is only valid
    // under the Causal Markov condition on the graph; every other
    // identifier in the crate (backdoor, IV, RD) records its structural
    // assumptions, and general ID must not be the exception.
    let id = IdIdentifier::new();
    let prep = id.prepare_dag(&chain_dag()).unwrap();
    let q = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
    let mut ws = IdentificationWorkspace::default();
    let res = id.identify_ate(&prep, &q, &mut ws).unwrap();
    assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
    assert!(
        res.required_assumptions
            .entries
            .iter()
            .any(|r| matches!(r.assumption, antecedent_core::Assumption::CausalMarkov)),
        "general ID must record the Causal Markov assumption it relies on"
    );
}

#[test]
fn hedge_not_identified() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../conformance/identify/id_hedge/expected.json"
    ))
    .unwrap();
    assert_eq!(fixture["cases"][1]["certificate"].as_str(), Some("hedge"));
    let external: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../conformance/identify/general_id_hedge/expected.json"
    ))
    .unwrap();
    assert_eq!(external["case"], "nonidentifiable_or_unidentified");
    assert_eq!(external["treatment"], "t");
    assert_eq!(external["outcome"], "y");
    assert_eq!(external["expected_status_family"], "unidentified");
    let oracle_estimand = external["reference"]["outputs"]["estimand"].as_str().unwrap();
    assert_eq!(oracle_estimand.matches("No such variable(s) found!").count(), 4);
    assert!(
        !oracle_estimand.contains("Estimand expression:"),
        "external oracle must refuse every standard identification strategy on the bow arc"
    );
    let (g, nodes) = admg_from_oracle_dot(external["graph_dot"].as_str().unwrap());
    let id = IdIdentifier::new();
    let prep = id.prepare(&g).unwrap();
    let q = AverageEffectQuery::binary_ate(
        nodes.id(external["treatment"].as_str().unwrap()),
        nodes.id(external["outcome"].as_str().unwrap()),
    );
    let mut ws = IdentificationWorkspace::default();
    let res = id.identify_ate(&prep, &q, &mut ws).unwrap();
    assert_eq!(res.status, IdentificationStatus::NotIdentified);
    assert!(res.diagnostics.iter().any(|d| d.code.as_ref() == "identify.hedge"));
}

#[test]
fn frontdoor_admg_identified() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../conformance/identify/general_id_frontdoor/expected.json"
    ))
    .unwrap();
    assert_eq!(fixture["expected_status_family"], "identified");
    let (g, nodes) = admg_from_oracle_dot(fixture["graph_dot"].as_str().unwrap());
    let id = IdIdentifier::new();
    let prep = id.prepare(&g).unwrap();
    let q = AverageEffectQuery::binary_ate(
        nodes.id(fixture["treatment"].as_str().unwrap()),
        nodes.id(fixture["outcome"].as_str().unwrap()),
    );
    let mut ws = IdentificationWorkspace::default();
    let res = id.identify_ate(&prep, &q, &mut ws).unwrap();
    assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
}

#[test]
fn soft_constant_and_shift_ate_reduce_to_set() {
    let id = IdIdentifier::new();
    let prep = id.prepare_dag(&chain_dag()).unwrap();
    let mut ws = IdentificationWorkspace::default();

    let soft = CausalQuery::AverageEffect(AverageEffectQuery::new(
        VariableId::from_raw(1),
        VariableId::from_raw(2),
        Arc::from([]),
        Intervention::set(VariableId::from_raw(1), Value::f64(0.0)),
        Intervention::soft(VariableId::from_raw(1), MechanismOverride::constant(1.0)),
        TargetPopulation::AllObserved,
    ));
    let res = id.identify(&prep, &soft, &mut ws).unwrap();
    assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);

    let shift = CausalQuery::AverageEffect(AverageEffectQuery::new(
        VariableId::from_raw(1),
        VariableId::from_raw(2),
        Arc::from([]),
        Intervention::set(VariableId::from_raw(1), Value::f64(0.0)),
        Intervention::shift(VariableId::from_raw(1), Value::f64(1.0)),
        TargetPopulation::AllObserved,
    ));
    let err = id.identify(&prep, &shift, &mut ws).unwrap_err();
    assert!(
        matches!(err, IdentificationError::UnsupportedQuery { message } if message.contains("Shift")),
        "{err}"
    );
}

#[test]
fn admg_observational_line1_uses_c_component_factorization() {
    // A ↔ B, no directed edges: Markov product would be P(A)P(B); Tian Q is P(A)P(B|A).
    let mut g = Admg::with_variables(2);
    g.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let id = IdIdentifier::new();
    let prep = id.prepare(&g).unwrap();
    let q = CausalQuery::Distribution(
        antecedent_core::InterventionalDistributionQuery::new(
            VariableId::from_raw(0),
            Arc::from([]),
        )
        .with_outcomes(Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)])),
    );
    let mut ws = IdentificationWorkspace::default();
    let res = id.identify(&prep, &q, &mut ws).unwrap();
    assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
    let functional = res.estimands[0].functional;
    assert!(
        distribution_has_nonempty_cond(&res.arena, functional),
        "line 1 on a bidirected ADMG must use Q-component factors, not ∏ P(vi)"
    );
}

fn distribution_has_nonempty_cond(arena: &CausalExprArena, id: ExprId) -> bool {
    match arena.node(id) {
        ExprNode::Distribution { conditioned_on, .. } => !arena.var_set(*conditioned_on).is_empty(),
        ExprNode::Product(list) => {
            arena.list(*list).iter().any(|&e| distribution_has_nonempty_cond(arena, e))
        }
        ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => {
            distribution_has_nonempty_cond(arena, *expr)
        }
        ExprNode::Ratio { numerator, denominator } => {
            distribution_has_nonempty_cond(arena, *numerator)
                || distribution_has_nonempty_cond(arena, *denominator)
        }
        _ => false,
    }
}

/// With ID complete, an `Err` is never a verdict on identifiability, so each remaining
/// one carries its own kind: a malformed query, and a broken internal invariant.
#[test]
fn remaining_errors_are_typed_by_cause() {
    let id = IdIdentifier::new();
    let prep = id.prepare_dag(&chain_dag()).unwrap();
    let y = VariableId::from_raw(2);
    let outcome_intervened =
        CausalQuery::Distribution(antecedent_core::InterventionalDistributionQuery::new(
            y,
            [Intervention::set(y, Value::f64(1.0))],
        ));
    let err = id
        .identify(&prep, &outcome_intervened, &mut IdentificationWorkspace::default())
        .unwrap_err();
    assert!(matches!(err, IdentificationError::InvalidQuery { .. }), "{err:?}");

    // A law that lacks a factor for a node of the requested C-component cannot arise
    // from the recursion; if it did, that is a defect, not an unsupported query.
    let nodes = prep.admg().node_count();
    let law = Law { sumset: BitSet::with_len(nodes), factors: Vec::new() };
    let mut s = BitSet::with_len(nodes);
    s.insert(DenseNodeId::from_raw(2));
    let err = law.conditionals(&prep, &s, &BitSet::full(nodes)).unwrap_err();
    assert!(matches!(err, IdentificationError::InvariantViolated { .. }), "{err:?}");
}

/// Napkin `W -> Z -> X -> Y`, `W <-> X`, `W <-> Y`: line 7 → line 2 → line 6.
/// The conditional of the marginalized C-factor must be emitted as
/// `N / ∑_y N` with the denominator summing the *same* numerator node, the
/// shape an exact evaluator recognizes as a conditional on a null event
/// when positivity fails (rather than an unlocated 0/0).
#[test]
fn napkin_conditional_is_a_ratio_over_its_own_marginal() {
    let mut g = Admg::with_variables(4);
    for (a, b) in [(0, 1), (1, 2), (2, 3)] {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    for (a, b) in [(0, 2), (0, 3)] {
        g.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let id = IdIdentifier::new();
    let prep = id.prepare(&g).unwrap();
    let q = CausalQuery::Distribution(antecedent_core::InterventionalDistributionQuery::new(
        VariableId::from_raw(3),
        [Intervention::set(VariableId::from_raw(2), Value::f64(1.0))],
    ));
    let mut ws = IdentificationWorkspace::default();
    let res = id.identify(&prep, &q, &mut ws).unwrap();
    assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
    let rules: Vec<&str> = res.derivation.steps.iter().map(|s| s.rule.as_ref()).collect();
    assert_eq!(
        rules,
        [
            "general.id",
            "general.id.line3",
            "general.id.line7",
            "general.id.line2",
            "general.id.line6"
        ]
    );
    let ExprNode::Ratio { numerator, denominator } =
        res.arena.node(res.estimands[0].functional).clone()
    else {
        panic!("expected a ratio, got {}", res.arena.pretty(res.estimands[0].functional));
    };
    let ExprNode::SumOut { variables, expr } = res.arena.node(denominator).clone() else {
        panic!("denominator must be a sum");
    };
    assert_eq!(expr, numerator, "denominator must marginalize the numerator node itself");
    assert_eq!(res.arena.var_set(variables), [VariableId::from_raw(3)]);
    // Numerator: ∑_w P(w) P(x | w, z) P(y | w, z, x).
    let ExprNode::SumOut { variables, expr } = res.arena.node(numerator).clone() else {
        panic!("numerator must sum out w");
    };
    assert_eq!(res.arena.var_set(variables), [VariableId::from_raw(0)]);
    let ExprNode::Product(list) = res.arena.node(expr).clone() else {
        panic!("numerator body must be the carried C-factor");
    };
    assert_eq!(res.arena.list(list).len(), 3);
}

/// `Z -> T`, `Z -> Y`, `T -> Y` and a four-point grid: the mean curve is one identified
/// intervention mean per grid level, not the first level standing for the whole curve.
#[test]
fn mean_curve_general_id_identifies_every_grid_level() {
    let id = IdIdentifier::new();
    let prep = id.prepare_dag(&chain_dag()).unwrap();
    let (t, y) = (VariableId::from_raw(1), VariableId::from_raw(2));
    let grid = [0.0, 0.5, 1.0, 2.0];
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: y,
        treatment: ContinuousDomain::new(t, GridSpec::Values(Arc::from(grid))),
    });
    let res =
        id.identify_response(&prep, &response, &mut IdentificationWorkspace::default()).unwrap();
    assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
    assert!(matches!(res.query, CausalQuery::Response(_)));
    assert_eq!(res.estimands.len(), grid.len());
    for estimand in &res.estimands {
        assert_eq!(estimand.method_kind().unwrap(), EstimandMethod::GeneralId);
        let ExprNode::Expectation { function, .. } = res.arena.node(estimand.functional) else {
            panic!("each grid level is the mean of Y");
        };
        assert_eq!(function.variable(), y);
    }
    // The level is part of the functional, so no two grid points share one.
    for (i, a) in res.estimands.iter().enumerate() {
        for b in &res.estimands[i + 1..] {
            assert_ne!(a.functional, b.functional);
        }
    }
}

/// `P(A | do(A = 1))` is the point mass at 1, not the observational `P(A)`: an outcome that
/// is also an intervention target is an invalid query for ID, whole or partial overlap.
#[test]
fn outcome_that_is_an_intervention_target_is_refused() {
    let id = IdIdentifier::new();
    let prep = id.prepare_dag(&chain_dag()).unwrap();
    let (t, y) = (VariableId::from_raw(1), VariableId::from_raw(2));
    let mut ws = IdentificationWorkspace::default();
    let whole = CausalQuery::Distribution(antecedent_core::InterventionalDistributionQuery::new(
        t,
        [Intervention::set(t, Value::f64(1.0))],
    ));
    let partial = CausalQuery::Distribution(
        antecedent_core::InterventionalDistributionQuery::new(
            y,
            [Intervention::set(t, Value::f64(1.0))],
        )
        .with_outcomes([y, t]),
    );
    for query in [whole, partial] {
        let err = id.identify(&prep, &query, &mut ws).unwrap_err();
        assert!(matches!(err, IdentificationError::InvalidQuery { .. }), "{err}");
    }
    // Disjoint outcome and intervention stay valid.
    let fine = CausalQuery::Distribution(antecedent_core::InterventionalDistributionQuery::new(
        y,
        [Intervention::set(t, Value::f64(1.0))],
    ));
    assert!(id.identify(&prep, &fine, &mut ws).is_ok());
}

#[test]
fn soft_linear_gaussian_still_unsupported() {
    let id = IdIdentifier::new();
    let prep = id.prepare_dag(&chain_dag()).unwrap();
    let mut ws = IdentificationWorkspace::default();
    let soft = CausalQuery::AverageEffect(AverageEffectQuery::new(
        VariableId::from_raw(1),
        VariableId::from_raw(2),
        Arc::from([]),
        Intervention::set(VariableId::from_raw(1), Value::f64(0.0)),
        Intervention::soft(
            VariableId::from_raw(1),
            MechanismOverride::named("linear_gaussian", vec![1.0, 0.0]),
        ),
        TargetPopulation::AllObserved,
    ));
    let err = id.identify(&prep, &soft, &mut ws).unwrap_err();
    assert!(
        matches!(err, IdentificationError::UnsupportedQuery { message } if message.contains("Soft")),
        "{err}"
    );
}

/// The counterfactual-identification hook charges every ID recursion: the
/// front-door `P_x(y)` charges each recursive call with its depth and live
/// bytes, and an error from the charge stops ID with that error.
#[test]
fn every_metered_recursion_is_charged_and_a_stop_ends_it() {
    let mut g = Admg::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    g.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    let prepared = PreparedAdmg::new(g).unwrap();
    let set = |ids: &[u32]| {
        let mut b = BitSet::with_len(3);
        for &i in ids {
            b.insert(DenseNodeId::from_raw(i));
        }
        b
    };
    // P_x(y): line 3 and the line-4 district factorization recurse.
    let (y, x) = (set(&[2]), set(&[0]));
    let mut charges = Vec::new();
    let mut record = |depth: usize, bytes: u64| {
        charges.push((depth, bytes));
        Ok(())
    };
    let outcome = identify_interventional_metered(
        &prepared,
        &y,
        &x,
        &mut GraphWorkspace::default(),
        &mut record,
    )
    .unwrap();
    assert!(matches!(outcome, MeteredId::Identified { .. }));
    assert!(charges.len() >= 3, "{charges:?}");
    assert_eq!(charges[0].0, 1, "the first call is at depth one");
    assert!(charges.iter().map(|c| c.0).max().unwrap() >= 2, "{charges:?}");
    assert!(charges.windows(2).all(|w| w[1].1 >= w[0].1), "live bytes only grow: {charges:?}");
    // A stop on the second charge ends ID with that error.
    let mut calls = 0;
    let mut stop = |_: usize, _: u64| {
        calls += 1;
        if calls >= 2 { Err(IdentificationError::Cancelled) } else { Ok(()) }
    };
    let stopped = identify_interventional_metered(
        &prepared,
        &y,
        &x,
        &mut GraphWorkspace::default(),
        &mut stop,
    );
    assert!(matches!(stopped, Err(IdentificationError::Cancelled)));
    assert_eq!(calls, 2);
    // The bow arc's P_x(y) is a hedge carrying its problem.
    let mut bow = Admg::with_variables(2);
    bow.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    bow.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let bow = PreparedAdmg::new(bow).unwrap();
    let (y, x) = (
        {
            let mut b = BitSet::with_len(2);
            b.insert(DenseNodeId::from_raw(1));
            b
        },
        {
            let mut b = BitSet::with_len(2);
            b.insert(DenseNodeId::from_raw(0));
            b
        },
    );
    let MeteredId::Hedge(hedge) = identify_interventional_metered(
        &bow,
        &y,
        &x,
        &mut GraphWorkspace::default(),
        &mut |_, _| Ok(()),
    )
    .unwrap() else {
        panic!("expected a hedge")
    };
    hedge.verify_carried().unwrap();
}

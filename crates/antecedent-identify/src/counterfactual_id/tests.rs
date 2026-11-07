use super::evaluation::{DenseJoint, NullExtension, TermEvaluator, factor_provider};
use super::*;

fn engine<'p, 'c>(
    problem: &'p CounterfactualIdProblem,
    ctx: &'c ExecutionContext,
) -> Engine<'p, 'c> {
    let budget = SearchBudget::with_memory(
        COUNTERFACTUAL_ID_DEFAULT_LIMITS,
        COUNTERFACTUAL_ID_MEMORY_BYTES,
        ctx,
    )
    .unwrap();
    Engine {
        problem,
        prepared: problem.prepared().unwrap(),
        parents: (0..problem.node_count())
            .map(|v| problem.parents(u32::try_from(v).unwrap()))
            .collect(),
        search: SharedSearch::new(budget),
        next_symbol: 0,
        retained_bytes: 0,
        first_graph: None,
        workspace: GraphWorkspace::default(),
    }
}

fn level(x: f64) -> CfValue {
    CfValue::Level(x.to_bits())
}

/// The effect-on-the-treated shape never reaches an ID hedge (the sweep in
/// `tests/counterfactual_id_search.rs` records none), so the hedge branch is
/// exercised on the engine directly: `P(Y_x = 1)` on the bow arc is one
/// district `{Y_x}` whose term `P_x(y)` ID refuses with a hedge.
#[test]
fn bidirected_self_loop_is_invalid_query() {
    let err = CounterfactualIdProblem::new(vec![vec![0.0, 1.0]; 2], &[], &[(0, 0)]).unwrap_err();
    assert_eq!(err.code, "invalid_argument");
    assert_eq!(err.detail, antecedent_core::reason_details::counterfactual_id::INVALID_QUERY);
    assert!(CounterfactualIdProblem::new(vec![vec![0.0, 1.0]; 2], &[], &[(0, 1)]).is_ok());
}

#[test]
fn an_id_hedge_on_a_district_term_is_a_verifiable_obstruction() {
    let problem =
        CounterfactualIdProblem::new(vec![vec![0.0, 1.0]; 2], &[(0, 1)], &[(0, 1)]).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let mut engine = engine(&problem, &ctx);
    let event = vec![Atom { var: 1, value: level(1.0), sub: BTreeMap::from([(0, level(1.0))]) }];
    let Err(Halt::Obstruction(boxed)) = engine.id_star(event, 0) else {
        panic!("expected an obstruction")
    };
    let obstruction = *boxed;
    let CounterfactualObstruction::Hedge { treatments, outcomes, f, f_prime, .. } = &obstruction
    else {
        panic!("expected a hedge")
    };
    assert_eq!((treatments.as_slice(), outcomes.as_slice()), (&[0][..], &[1][..]));
    assert_eq!((f.as_slice(), f_prime.as_slice()), (&[0, 1][..], &[1][..]));
    obstruction.verify().unwrap();
    let refusal = CounterfactualIdRefusal::hedge(obstruction.clone());
    assert_eq!(
        (refusal.code, refusal.detail),
        (
            "route_not_supported",
            antecedent_core::reason_details::counterfactual_id::COUNTERFACTUAL_HEDGE
        )
    );
    // A tampered hedge fails the definition check.
    let mut tampered = obstruction;
    if let CounterfactualObstruction::Hedge { f_prime, .. } = &mut tampered {
        *f_prime = vec![0];
    }
    assert!(tampered.verify().is_err());
}

fn run(
    problem: &CounterfactualIdProblem,
    event: Vec<Atom>,
) -> Result<CounterfactualFunctional, Halt> {
    let ctx = ExecutionContext::for_tests(1);
    engine(problem, &ctx).id_star(event, 0)
}

fn atom(var: u32, value: f64, sub: &[(u32, f64)]) -> Atom {
    Atom { var, value: level(value), sub: sub.iter().map(|&(v, x)| (v, level(x))).collect() }
}

fn conflict_of(result: Result<CounterfactualFunctional, Halt>) -> (u32, Option<CfValue>) {
    let Err(Halt::Obstruction(boxed)) = result else { panic!("expected an obstruction") };
    boxed.verify().unwrap();
    let CounterfactualObstruction::ConflictingSubscripts { variable, event, .. } = *boxed else {
        panic!("expected a conflict")
    };
    (variable, event)
}

/// The guards the effect-on-the-treated shape cannot reach (its top-level
/// counterfactual graph has at most one copy per variable and one
/// intervened world), each reached by a synthetic conjunction on the engine.
#[test]
fn synthetic_conjunctions_reach_the_guards_the_ett_shape_cannot() {
    let binary = |n| vec![vec![0.0, 1.0]; n];
    // X -> Y, no confounding: {Y_{x=0} = 0, Y_{x=1} = 1}. The two copies of Y
    // see fixed parents at two levels, so Lemma 24 does not merge them (a
    // merge would read one node at two levels: an exact zero), and the copies
    // share Y's exogenous term, so they form one district (split, the two
    // terms would multiply as if independent). ID* fails on X.
    let chain = CounterfactualIdProblem::new(binary(2), &[(0, 1)], &[]).unwrap();
    let event = vec![atom(1, 0.0, &[(0, 0.0)]), atom(1, 1.0, &[(0, 1.0)])];
    assert_eq!(conflict_of(run(&chain, event)).0, 0);
    // X -> Y, X -> W, X <-> Y, W <-> Y: {Y_{x=1} = 1, W = 0} keeps X as a free
    // ancestor of W in the district that sets X: a free node conflicts.
    let free =
        CounterfactualIdProblem::new(binary(3), &[(0, 1), (0, 2)], &[(0, 1), (1, 2)]).unwrap();
    let event = vec![atom(1, 1.0, &[(0, 1.0)]), atom(2, 0.0, &[])];
    assert_eq!(conflict_of(run(&free, event)), (0, None));
    // P -> V, P <-> V: {V_{p=1} = 0, V = 1, P = 1}. The copies of V stay
    // unmerged (a node parent against a fixed one), no subscript conflicts
    // (P is read at the level it is set to), and by consistency V_{p=1} = V
    // when P = 1: an exact zero.
    let pv = CounterfactualIdProblem::new(binary(2), &[(0, 1)], &[(0, 1)]).unwrap();
    let event = vec![atom(1, 0.0, &[(0, 1.0)]), atom(1, 1.0, &[]), atom(0, 1.0, &[])];
    assert!(matches!(run(&pv, event), Ok(CounterfactualFunctional::Zero)));
    // X -> Y and an isolated W: W_{x=1} and W merge (no parents), so reading
    // the one node at two levels is an exact zero.
    let isolated = CounterfactualIdProblem::new(binary(3), &[(0, 1)], &[]).unwrap();
    let event = vec![atom(2, 0.0, &[(0, 1.0)]), atom(2, 1.0, &[])];
    assert!(matches!(run(&isolated, event), Ok(CounterfactualFunctional::Zero)));
}

/// A term `sum_m P(m | x) P(y | m)` with `P(X = x) = 0` and a three-level `M`:
/// its value is `sum_m q_m g(m)` for the extension `q` of `P(M | X = x)`,
/// with `g = P(y = 1 | m) = (0.5, 0.25, 0.75)`. The uniform extension and all
/// mass on the first level agree (0.5); all mass on level 1 gives 0.25, so
/// the term is refused as a positivity violation.
#[test]
fn a_summed_three_level_conditional_on_a_zero_mass_event_is_caught() {
    let (x, m, y) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
    let mut arena = CausalExprArena::new();
    let none = arena.empty_intervention_set();
    let (xs, ms, ys) =
        (arena.intern_var_set([x]), arena.intern_var_set([m]), arena.intern_var_set([y]));
    let m_given_x = arena.intern_distribution(ms, xs, none, DomainRef::Observational);
    let y_given_m = arena.intern_distribution(ys, ms, none, DomainRef::Observational);
    let list = arena.intern_list([m_given_x, y_given_m]);
    let product = arena.intern(ExprNode::Product(list));
    let expression = arena.intern(ExprNode::SumOut { variables: ms, expr: product });
    let term = CfTerm {
        intervention: vec![(0, level(1.0))],
        event: vec![(2, level(1.0))],
        arena,
        expression,
    };
    // X = 0 always; P(m) = (0.3, 0.3, 0.4); P(y = 1 | m) = g(m). Row-major (x, m, y).
    let g = [0.5, 0.25, 0.75];
    let mut probabilities = vec![0.0; 12];
    for (level_m, (share, g)) in [0.3, 0.3, 0.4].iter().zip(g).enumerate() {
        probabilities[level_m * 2] = share * (1.0 - g);
        probabilities[level_m * 2 + 1] = share * g;
    }
    let joint = DenseJoint {
        levels: vec![vec![0.0, 1.0], vec![0.0, 1.0, 2.0], vec![0.0, 1.0]],
        probabilities,
    };
    let assignment = Assignment::from_pairs([(x, Value::f64(1.0)), (y, Value::f64(1.0))]);
    let compiled = term.arena.compile(term.expression).unwrap();
    let under = |fill| {
        let (provider, touched_null) = factor_provider(&joint, &term.arena, term.expression, fill);
        assert!(touched_null);
        compiled
            .evaluate_with(&term.arena, &provider, &EvalContext::default(), &assignment)
            .unwrap()
    };
    assert!((under(NullExtension::Uniform) - 0.5).abs() < 1e-12);
    assert!((under(NullExtension::Level(0)) - 0.5).abs() < 1e-12);
    assert!((under(NullExtension::Level(1)) - 0.25).abs() < 1e-12);
    let ctx = ExecutionContext::for_tests(1);
    let refusal = TermEvaluator::new(&joint).term(&term, &HashMap::new(), &ctx).unwrap_err();
    assert_eq!(
        refusal.detail,
        antecedent_core::reason_details::counterfactual_id::POSITIVITY_VIOLATION
    );
}

#[test]
fn a_context_cancelled_mid_decision_stops_with_a_receipt() {
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
    let shape = classify(&problem, &query).unwrap();
    let given = vec![(0, 0.0f64.to_bits())];
    let ctx = ExecutionContext::for_tests(1);
    let full = decide_under(
        &problem,
        &query,
        shape,
        given.clone(),
        COUNTERFACTUAL_ID_DEFAULT_LIMITS,
        SharedSearch::new(SearchBudget::new(COUNTERFACTUAL_ID_DEFAULT_LIMITS, &ctx).unwrap()),
    )
    .unwrap();
    let total = full.search.operations_consumed;
    // Cancel after the first, a middle and the last-but-one charge: each stops there.
    for after in [1, total / 2, total - 1] {
        let ctx = ExecutionContext::for_tests(1);
        let mut search =
            SharedSearch::new(SearchBudget::new(COUNTERFACTUAL_ID_DEFAULT_LIMITS, &ctx).unwrap());
        search.cancel_after = Some((after, ctx.cancellation.clone()));
        let refusal = decide_under(
            &problem,
            &query,
            shape,
            given.clone(),
            COUNTERFACTUAL_ID_DEFAULT_LIMITS,
            search,
        )
        .unwrap_err();
        let receipt = refusal.receipt.expect("a receipt");
        assert_eq!(receipt.stop, SearchStop::Cancelled);
        assert_eq!(receipt.operations_consumed, Some(after));
        assert_eq!(refusal.detail, antecedent_core::reason_details::counterfactual_id::BUDGET);
    }
}

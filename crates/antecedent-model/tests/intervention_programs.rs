//! Programmable interventions: value expressions compiled into overlays and executed by
//! the shared column kernel, with dependency validation and per-route licensing.
#![allow(
    clippy::float_cmp,
    reason = "the kernel is bit-identical to the interpreter and to hand arithmetic, so equality is exact"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use antecedent_core::{
    BatchCallback, CallbackError, CallbackImplementation, CallbackOutput, CallbackSpec, CausalRng,
    ExecutionContext, ExpressionFault, Intervention, InterventionSequence, PinnedDependency,
    PolicyReads, Portability, SequencedIntervention, TemporalPolicy, Value, ValueExpr, VariableId,
    reason_code,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, InterventionOverlay, MechanismSlot,
    MechanismWorkspace, ModelError, ModelView, PolicyRoute, ValueBatch, policy_support,
    sample_interventional, sample_observational, sample_with_overlay,
};

fn var(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn model(n: u32, edges: &[(u32, u32)], slots: Vec<MechanismSlot>) -> CompiledCausalModel {
    let mut graph = Dag::with_variables(n);
    for &(from, to) in edges {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    CompiledCausalModel::compile(graph)
        .unwrap()
        .with_mechanisms(CompiledMechanismStore { slots: Arc::from(slots) })
}

fn gaussian(intercept: f64, coeffs: &[f64], sigma: f64) -> MechanismSlot {
    MechanismSlot::LinearGaussian { intercept, coeffs: Arc::from(coeffs.to_vec()), sigma }
}

/// price(0) -> demand(1) -> revenue(2), price -> revenue; the second revenue parent is
/// whichever the compiled gather lists second, so coefficients are aligned by lookup.
fn pricing_chain() -> (CompiledCausalModel, [f64; 2]) {
    let edges = [(0, 1), (0, 2), (1, 2)];
    let probe = model(3, &edges, vec![MechanismSlot::Vacant; 3]);
    let parents = probe.gather_for(DenseNodeId::from_raw(2)).unwrap().parents.clone();
    // revenue = 3 + 0.5 * price + 2 * demand + noise
    let by_parent = |dense: &DenseNodeId| if dense.as_usize() == 0 { 0.5 } else { 2.0 };
    let coeffs = [by_parent(&parents[0]), by_parent(&parents[1])];
    let slots =
        vec![gaussian(11.0, &[], 2.0), gaussian(50.0, &[-0.8], 1.0), gaussian(3.0, &coeffs, 0.5)];
    (model(3, &edges, slots), [0.5, 2.0])
}

fn column(batch: &ValueBatch, node: usize) -> Vec<f64> {
    batch.column(node).unwrap().to_vec()
}

fn draw(model: &CompiledCausalModel, interventions: &[Intervention], n: usize) -> ValueBatch {
    let mut rng = CausalRng::from_seed(20_260);
    let mut ws = MechanismWorkspace::default();
    sample_interventional(
        model,
        interventions,
        n,
        &mut rng,
        &mut ws,
        &ExecutionContext::for_tests(1),
    )
    .unwrap()
}

fn observe(model: &CompiledCausalModel, n: usize) -> ValueBatch {
    let mut rng = CausalRng::from_seed(20_260);
    let mut ws = MechanismWorkspace::default();
    sample_observational(model, n, &mut rng, &mut ws, &ExecutionContext::for_tests(1)).unwrap()
}

/// `(code, detail)` of a coded policy refusal.
fn refusal(error: &ModelError) -> (String, String) {
    let text = error.to_string();
    let body = text.strip_prefix("unsupported: ").unwrap_or(&text);
    let (code, tail) =
        antecedent_core::reason_code::split_prefix(body).unwrap_or_else(|| panic!("{text}"));
    (code.to_string(), tail.split(':').next().unwrap().to_string())
}

fn expect_refusal(
    model: &CompiledCausalModel,
    interventions: &[Intervention],
    code: &str,
    detail: &str,
) {
    let error = InterventionOverlay::from_interventions(model, interventions)
        .expect_err("the policy must be refused");
    assert_eq!(refusal(&error), (code.to_string(), detail.to_string()), "{error}");
    // The sampler refuses identically and produces no draws.
    let mut rng = CausalRng::from_seed(1);
    let mut ws = MechanismWorkspace::default();
    let sampled = sample_interventional(
        model,
        interventions,
        8,
        &mut rng,
        &mut ws,
        &ExecutionContext::for_tests(1),
    );
    assert_eq!(refusal(&sampled.expect_err("no draws")), (code.to_string(), detail.to_string()));
}

#[test]
fn a_price_floor_holds_for_every_realized_price_and_flows_downstream_on_the_same_noise() {
    let (model, [price_to_revenue, demand_to_revenue]) = pricing_chain();
    let n = 50_000;
    let floor =
        ValueExpr::constant(10.0).max(ValueExpr::constant(0.85).mul(ValueExpr::generated()));
    let policy = [Intervention::transform(var(0), floor)];

    let base = observe(&model, n);
    let after = draw(&model, &policy, n);
    let (p0, d0, r0) = (column(&base, 0), column(&base, 1), column(&base, 2));
    let (p1, d1, r1) = (column(&after, 0), column(&after, 1), column(&after, 2));

    // Independent hand simulation. The transform draws no randomness, so the same seed
    // gives the same noise; each downstream value moves by its mechanism's response to the
    // upstream change and nothing else.
    let mut held = 0usize;
    for i in 0..n {
        let expected = 10.0_f64.max(0.85 * p0[i]);
        assert_eq!(p1[i].to_bits(), expected.to_bits(), "row {i}");
        assert!(p1[i] >= 10.0);
        held += usize::from(p1[i] == 10.0);
        let demand = d0[i] - 0.8 * (p1[i] - p0[i]);
        assert!((d1[i] - demand).abs() < 1e-9, "demand row {i}");
        let revenue =
            r0[i] + price_to_revenue * (p1[i] - p0[i]) + demand_to_revenue * (d1[i] - d0[i]);
        assert!((r1[i] - revenue).abs() < 1e-9, "revenue row {i}");
    }
    let independent_count = p0.iter().filter(|p| 0.85 * **p < 10.0).count();
    assert_eq!(held, independent_count);
    assert!(
        held > n / 10 && held < n * 9 / 10,
        "the floor must bind on a non-trivial share of draws, got {held} of {n}"
    );
}

#[test]
fn a_transform_keeps_the_noise_and_incoming_edges_while_an_assignment_replaces_them() {
    // z -> x -> y
    let slots =
        vec![gaussian(0.0, &[], 1.0), gaussian(0.0, &[0.7], 0.5), gaussian(1.0, &[-2.0], 0.3)];
    let model = model(3, &[(0, 1), (1, 2)], slots);
    let n = 20_000;
    let base = observe(&model, n);
    let (z0, x0, y0) = (column(&base, 0), column(&base, 1), column(&base, 2));

    let shifted = draw(
        &model,
        &[Intervention::transform(var(1), ValueExpr::generated().add(ValueExpr::constant(1.0)))],
        n,
    );
    let assigned = draw(&model, &[Intervention::assign(var(1), ValueExpr::constant(1.0))], n);
    let (xt, yt) = (column(&shifted, 1), column(&shifted, 2));
    let (xa, ya) = (column(&assigned, 1), column(&assigned, 2));

    for i in 0..n {
        // Transform: x' = x + 1 keeps its dependence on z and its own noise; y moves by
        // exactly the structural slope times the shift.
        assert!((xt[i] - (x0[i] + 1.0)).abs() < 1e-12);
        assert!((yt[i] - (y0[i] - 2.0)).abs() < 1e-9);
        // Assignment: x' ignores z and its noise; y keeps only its own noise.
        assert_eq!(xa[i], 1.0);
    }
    // An assignment draws no noise for x, so y's disturbance comes from a different place in
    // the stream than in the observational run: compare moments, not rows.
    let mean_ya = ya.iter().sum::<f64>() / n as f64;
    assert!((mean_ya - (1.0 - 2.0)).abs() < 0.01, "{mean_ya}");
    let corr = |a: &[f64], b: &[f64]| {
        let (ma, mb) = (a.iter().sum::<f64>() / n as f64, b.iter().sum::<f64>() / n as f64);
        let cov: f64 = a.iter().zip(b).map(|(p, q)| (p - ma) * (q - mb)).sum();
        let va: f64 = a.iter().map(|p| (p - ma).powi(2)).sum();
        let vb: f64 = b.iter().map(|q| (q - mb).powi(2)).sum();
        cov / (va * vb).sqrt()
    };
    assert!(corr(&z0, &xt) > 0.5, "a transform keeps the edge from z");
    assert!(xa.iter().all(|v| *v == 1.0), "an assignment severs it");
    assert!(corr(&z0, &ya).abs() < 0.03, "y no longer depends on z");
    let variance = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / n as f64;
        v.iter().map(|p| (p - m).powi(2)).sum::<f64>() / n as f64
    };
    // Var(y | transform) = 4 Var(x) + 0.09; Var(y | assignment) = 0.09.
    assert!((variance(&yt) - variance(&y0)).abs() < 1e-9);
    assert!(variance(&ya) < 0.12 && variance(&yt) > 1.0);
}

#[test]
fn the_batch_kernel_equals_the_scalar_interpreter_bit_for_bit() {
    // z(0) -> x(1) -> w(2)
    let slots =
        vec![gaussian(0.0, &[], 1.0), gaussian(0.5, &[1.0], 1.0), gaussian(0.0, &[1.0], 1.0)];
    let model = model(3, &[(0, 1), (1, 2)], slots);
    let n = 5_000;
    let base = observe(&model, n);
    let (z0, x0) = (column(&base, 0), column(&base, 1));
    let z = var(0);
    let g = ValueExpr::generated;
    let c = ValueExpr::constant;
    let cases: Vec<(&str, ValueExpr)> = vec![
        ("affine", c(2.0).mul(g()).add(c(1.0))),
        ("difference of a read", g().sub(ValueExpr::var(z))),
        ("explicit read", c(0.5).mul(ValueExpr::var(z)).add(g())),
        ("conditional", ValueExpr::var(z).gt(c(0.0)).if_then_else(g().add(c(2.0)), g())),
        ("clip", g().clip(c(-1.0), c(1.0))),
        ("indicator", g().gt(c(0.0)).and(ValueExpr::var(z).lt(c(0.5))).indicator()),
        ("negated disjunction", g().lt(c(-1.0)).or(g().gt(c(1.0))).not().indicator().neg()),
        ("guarded quotient", g().equals(c(0.0)).if_then_else(c(0.0), c(1.0).div(g()))),
        ("min and max", g().min(ValueExpr::var(z)).max(c(-0.25))),
    ];
    for (name, expr) in cases {
        let after = draw(&model, &[Intervention::transform(var(1), expr.clone())], n);
        let x1 = column(&after, 1);
        let repeat = draw(&model, &[Intervention::transform(var(1), expr.clone())], n);
        assert_eq!(column(&repeat, 2), column(&after, 2), "{name}: the same seed repeats");
        for i in 0..n {
            let expected = expr
                .evaluate(Some(x0[i]), &|v| (v == z).then_some(z0[i]))
                .unwrap_or_else(|| panic!("{name}: interpreter"));
            assert_eq!(x1[i].to_bits(), expected.to_bits(), "{name}, row {i}");
        }
    }
}

#[test]
fn clipping_bounds_every_draw_and_the_clipped_mass_is_the_tail_probability() {
    let slots = vec![gaussian(0.0, &[], 1.0), gaussian(0.0, &[], 1.0)];
    let model = model(2, &[], slots);
    let n = 40_000;
    let base = observe(&model, n);
    let x0 = column(&base, 1);
    let clipped = draw(
        &model,
        &[Intervention::transform(
            var(1),
            ValueExpr::generated().clip(ValueExpr::constant(-1.0), ValueExpr::constant(1.0)),
        )],
        n,
    );
    let xc = column(&clipped, 1);
    assert!(xc.iter().all(|v| (-1.0..=1.0).contains(v)));
    for i in 0..n {
        assert_eq!(xc[i], x0[i].clamp(-1.0, 1.0));
    }
    // P(|x| > 1) = 2 (1 - Phi(1)) = 0.3173
    let mass = xc.iter().filter(|v| v.abs() == 1.0).count() as f64 / n as f64;
    assert!((mass - 0.3173).abs() < 0.01, "clipped mass {mass}");
}

#[test]
fn a_conditional_branch_selects_per_row_from_the_sampled_condition() {
    // The zero-weight edge z -> x fixes the evaluation order, so the program's read of z
    // does not reorder the noise draws the observational run made.
    let slots = vec![gaussian(0.0, &[], 1.0), gaussian(0.0, &[0.0], 1.0)];
    let model = model(2, &[(0, 1)], slots);
    let n = 40_000;
    let base = observe(&model, n);
    let (z0, x0) = (column(&base, 0), column(&base, 1));
    let branched = draw(
        &model,
        &[Intervention::transform(
            var(1),
            ValueExpr::var(var(0)).gt(ValueExpr::constant(0.0)).if_then_else(
                ValueExpr::generated().add(ValueExpr::constant(2.0)),
                ValueExpr::generated(),
            ),
        )],
        n,
    );
    let xb = column(&branched, 1);
    for i in 0..n {
        let expected = if z0[i] > 0.0 { x0[i] + 2.0 } else { x0[i] };
        assert_eq!(xb[i], expected, "row {i}");
    }
}

#[test]
fn policies_are_evaluated_in_dependency_order_not_declaration_or_node_order() {
    // No edges: b (node 0) would run before a (node 1) in the model's own order, yet
    // b := 2 * a must see the transformed a.
    let slots =
        vec![MechanismSlot::Constant { value: 0.0 }, MechanismSlot::Constant { value: 3.0 }];
    let model = model(2, &[], slots);
    let a = var(1);
    let b = var(0);
    let policies = [
        Intervention::assign(b, ValueExpr::constant(2.0).mul(ValueExpr::var(a))),
        Intervention::transform(a, ValueExpr::generated().add(ValueExpr::constant(1.0))),
    ];
    let overlay = InterventionOverlay::from_interventions(&model, &policies).unwrap();
    assert!(overlay.order.is_some(), "reads add an edge, so the order is derived");
    let batch = sample_with_overlay(
        &ModelView::with_overlay(&model, overlay),
        16,
        &mut CausalRng::from_seed(3),
        &mut MechanismWorkspace::default(),
    )
    .unwrap();
    assert!(column(&batch, 1).iter().all(|v| *v == 4.0));
    assert!(column(&batch, 0).iter().all(|v| *v == 8.0));
    // Declaring the policies in the other order changes nothing.
    let reversed = [policies[1].clone(), policies[0].clone()];
    let again = draw(&model, &reversed, 16);
    assert_eq!(column(&again, 0), column(&batch, 0));
}

#[test]
fn circular_dependencies_are_refused_naming_the_cycle_before_any_draw() {
    let flat = model(2, &[], vec![MechanismSlot::Constant { value: 0.0 }; 2]);
    let (a, b) = (var(0), var(1));
    let circular = reason_code!("policy_dependency_circular");

    // a := b + 1 and b := a * 2 depend on each other.
    expect_refusal(
        &flat,
        &[
            Intervention::assign(a, ValueExpr::var(b).add(ValueExpr::constant(1.0))),
            Intervention::assign(b, ValueExpr::var(a).mul(ValueExpr::constant(2.0))),
        ],
        circular,
        "intervention_program.dependency_circular",
    );
    // The refusal names the cycle and the policies in it.
    let error = InterventionOverlay::from_interventions(
        &flat,
        &[Intervention::assign(a, ValueExpr::var(b)), Intervention::assign(b, ValueExpr::var(a))],
    )
    .unwrap_err();
    let text = error.to_string();
    assert!(text.contains(" -> ") && text.contains("policies on"), "{text}");
    assert!(text.contains("condensation"), "{text}");
    assert!(text.contains("no cyclic license"), "{text}");

    // Reading a descendant of the target closes a cycle through the graph.
    let edge = model(2, &[(0, 1)], vec![MechanismSlot::Constant { value: 0.0 }; 2]);
    expect_refusal(
        &edge,
        &[Intervention::assign(a, ValueExpr::var(b))],
        circular,
        "intervention_program.dependency_circular",
    );
    // A node cannot read itself; the generated value is what a transform reads.
    expect_refusal(
        &edge,
        &[Intervention::transform(a, ValueExpr::var(a))],
        circular,
        "intervention_program.dependency_circular",
    );
    // The same node reading its parent is fine.
    InterventionOverlay::from_interventions(&edge, &[Intervention::assign(b, ValueExpr::var(a))])
        .unwrap();
}

#[test]
fn unavailable_dependencies_are_refused_before_any_draw() {
    let flat = model(2, &[], vec![MechanismSlot::Constant { value: 0.0 }; 2]);
    let (a, b) = (var(0), var(1));
    let unavailable = reason_code!("policy_dependency_unavailable");

    // A variable that is not in the model.
    expect_refusal(
        &flat,
        &[Intervention::assign(a, ValueExpr::var(var(99)).add(ValueExpr::constant(1.0)))],
        unavailable,
        "intervention_program.variable_not_in_model",
    );
    // An assignment replaces the generated value, so it cannot read it.
    expect_refusal(
        &flat,
        &[Intervention::assign(a, ValueExpr::generated())],
        unavailable,
        "intervention_program.generated_value_unavailable",
    );
    // Reads must be declared exactly.
    let undeclared = Intervention::Assign {
        variable: a,
        expr: ValueExpr::var(b),
        reads: PolicyReads::default(),
    };
    expect_refusal(&flat, &[undeclared], unavailable, "intervention_program.undeclared_read");
}

#[test]
fn conflicting_directives_on_one_node_are_refused_but_identical_repeats_are_not() {
    let edge = model(2, &[(0, 1)], vec![gaussian(0.0, &[], 1.0), gaussian(0.0, &[1.0], 1.0)]);
    let x = var(1);
    let conflict = reason_code!("policy_directive_conflict");
    let detail = "intervention_program.directive_conflict";
    let bump = || ValueExpr::generated().add(ValueExpr::constant(1.0));
    let clip = ValueExpr::generated().clip(ValueExpr::constant(0.0), ValueExpr::constant(1.0));

    expect_refusal(
        &edge,
        &[Intervention::transform(x, bump()), Intervention::transform(x, clip)],
        conflict,
        detail,
    );
    expect_refusal(
        &edge,
        &[Intervention::transform(x, bump()), Intervention::assign(x, ValueExpr::constant(1.0))],
        conflict,
        detail,
    );
    expect_refusal(
        &edge,
        &[Intervention::transform(x, bump()), Intervention::set(x, Value::f64(1.0))],
        conflict,
        detail,
    );
    expect_refusal(
        &edge,
        &[Intervention::transform(x, bump()), Intervention::shift(x, Value::f64(1.0))],
        conflict,
        detail,
    );
    expect_refusal(
        &edge,
        &[
            Intervention::assign(x, ValueExpr::constant(1.0)),
            Intervention::stochastic(x, antecedent_core::StochasticPolicy::gaussian(0.0, 1.0)),
        ],
        conflict,
        detail,
    );
    // The same program stated twice is one program.
    InterventionOverlay::from_interventions(
        &edge,
        &[Intervention::transform(x, bump()), Intervention::transform(x, bump())],
    )
    .unwrap();
    // A transform acts on whatever the stochastic policy generated.
    let overlay = InterventionOverlay::from_interventions(
        &edge,
        &[
            Intervention::transform(x, bump()),
            Intervention::stochastic(x, antecedent_core::StochasticPolicy::gaussian(0.0, 1.0)),
        ],
    )
    .unwrap();
    assert!(overlay.program_at(1).is_some() && overlay.stochastic[1].is_some());
}

#[test]
fn output_types_are_checked_against_the_expression_and_the_node() {
    let slots = vec![
        gaussian(0.0, &[], 1.0),
        MechanismSlot::Discrete {
            support: Arc::from([0.0, 1.0]),
            probs: Arc::from([0.5, 0.5]),
            logit_coeffs: None,
        },
    ];
    let model = model(2, &[], slots);
    let (x, t) = (var(0), var(1));
    let output = reason_code!("policy_output_type_invalid");

    // A comparison is a boolean; it needs an explicit indicator to be a node value.
    expect_refusal(
        &model,
        &[Intervention::transform(x, ValueExpr::generated().gt(ValueExpr::constant(0.0)))],
        output,
        "intervention_program.output_type",
    );
    // An operand of the wrong type is a malformed expression.
    expect_refusal(
        &model,
        &[Intervention::assign(
            x,
            ValueExpr::constant(1.0).add(ValueExpr::var(t).gt(ValueExpr::constant(0.0))),
        )],
        reason_code!("invalid_argument"),
        "intervention_program.operand_type",
    );
    // A categorical node takes a level of its support, or a 0/1 indicator when the
    // support holds both, and nothing else.
    expect_refusal(
        &model,
        &[Intervention::assign(t, ValueExpr::var(x).mul(ValueExpr::constant(2.0)))],
        output,
        "intervention_program.categorical_output",
    );
    expect_refusal(
        &model,
        &[Intervention::assign(t, ValueExpr::constant(5.0))],
        output,
        "intervention_program.categorical_output",
    );
    let flag = ValueExpr::var(x).gt(ValueExpr::constant(0.0)).indicator();
    let batch = draw(&model, &[Intervention::assign(t, flag)], 1_000);
    let base = observe(&model_with_vacant_treatment(), 1_000);
    let (x_col, t_col) = (column(&batch, 0), column(&batch, 1));
    assert!(t_col.iter().zip(&x_col).all(|(t, x)| *t == f64::from(u8::from(*x > 0.0))));
    assert_eq!(column(&base, 0), x_col, "the assignment draws no randomness of its own");
}

fn model_with_vacant_treatment() -> CompiledCausalModel {
    model(2, &[], vec![gaussian(0.0, &[], 1.0), MechanismSlot::Constant { value: 0.0 }])
}

#[test]
fn history_reads_need_a_temporal_route_with_a_window() {
    let flat = model(2, &[], vec![MechanismSlot::Constant { value: 0.0 }; 2]);
    let history = reason_code!("policy_history_unavailable");
    expect_refusal(
        &flat,
        &[Intervention::assign(var(0), ValueExpr::lag(var(1), 1))],
        history,
        "intervention_program.history_unavailable",
    );
    // A look-back of zero steps is not history, and a future value does not exist.
    expect_refusal(
        &flat,
        &[Intervention::assign(var(0), ValueExpr::lag(var(1), 0))],
        history,
        "intervention_program.history_not_past",
    );
}

#[test]
fn non_finite_results_and_cancellation_stop_the_run_without_draws() {
    let model = model(1, &[], vec![gaussian(0.0, &[], 1.0)]);
    let x = var(0);
    let mut rng = CausalRng::from_seed(5);
    let mut ws = MechanismWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    // 1 / x is infinite only if a draw is exactly zero, so force the failure explicitly.
    let by_zero = Intervention::transform(x, ValueExpr::generated().div(ValueExpr::constant(0.0)));
    let error = sample_interventional(&model, &[by_zero], 10, &mut rng, &mut ws, &ctx).unwrap_err();
    assert!(matches!(error, ModelError::Numerical { .. }), "{error}");

    let cancelled = ExecutionContext::for_tests(1);
    cancelled.cancellation.cancel();
    let bump = Intervention::transform(x, ValueExpr::generated().add(ValueExpr::constant(1.0)));
    let error =
        sample_interventional(&model, &[bump], 10, &mut rng, &mut ws, &cancelled).unwrap_err();
    assert_eq!(error, ModelError::Cancelled);
}

#[test]
fn each_route_is_licensed_separately() {
    let (model, _) = pricing_chain();
    let floor = Intervention::transform(
        var(0),
        ValueExpr::constant(10.0).max(ValueExpr::constant(0.85).mul(ValueExpr::generated())),
    );
    let support = policy_support(&model, std::slice::from_ref(&floor));
    assert!(support.decision(PolicyRoute::Simulation).is_licensed());
    assert!(support.decision(PolicyRoute::CounterfactualExecution).is_licensed());
    let identification = support.decision(PolicyRoute::ObservationalIdentification);
    match identification {
        antecedent_model::RouteDecision::Refused { code, detail, .. } => {
            assert_eq!(code, reason_code!("policy_route_not_licensed"));
            assert_eq!(detail, "intervention_program.identification_requires_hard_level");
        }
        other @ (antecedent_model::RouteDecision::Licensed
        | antecedent_model::RouteDecision::LicensedUnder { .. }) => {
            panic!("a transform has no observational identification, got {other:?}")
        }
    }
    let antecedent_model::RouteDecision::Refused { message, .. } = identification else {
        unreachable!()
    };
    assert!(
        message.contains("observational identification") && message.contains("simulate"),
        "the refusal names the route and the licensed ones: {message}"
    );

    // A computed assignment that folds to a constant is a hard level on every route.
    let level =
        Intervention::assign(var(0), ValueExpr::constant(2.0).mul(ValueExpr::constant(6.0)));
    let support = policy_support(&model, &[level]);
    assert!(support.simulation.is_licensed() && support.counterfactual.is_licensed());
    assert!(support.identification.is_licensed());
    // An assignment that reads another variable is a function, not a level.
    let function = Intervention::assign(var(1), ValueExpr::var(var(0)));
    assert!(!policy_support(&model, &[function]).identification.is_licensed());
    assert!(
        policy_support(&model, &[Intervention::set(var(0), Value::f64(1.0))])
            .identification
            .is_licensed()
    );

    // An invalid program is licensed on no route, with the same coded reason.
    let flat = model_with_vacant_treatment();
    let bad = [
        Intervention::assign(var(0), ValueExpr::var(var(1))),
        Intervention::assign(var(1), ValueExpr::var(var(0))),
    ];
    let support = policy_support(&flat, &bad);
    for route in [
        PolicyRoute::Simulation,
        PolicyRoute::CounterfactualExecution,
        PolicyRoute::ObservationalIdentification,
    ] {
        match support.decision(route) {
            antecedent_model::RouteDecision::Refused { code, .. } => {
                assert_eq!(code, reason_code!("policy_dependency_circular"));
            }
            other @ (antecedent_model::RouteDecision::Licensed
            | antecedent_model::RouteDecision::LicensedUnder { .. }) => {
                panic!("{route:?}: {other:?}")
            }
        }
    }
}

#[test]
fn the_conditional_sampler_runs_programs_and_refuses_to_condition_on_them() {
    use antecedent_model::{ConditionEvidence, sample_conditional_interventional};
    // z -> x -> y; condition on y, transform x.
    let slots =
        vec![gaussian(0.0, &[], 1.0), gaussian(0.0, &[0.8], 0.5), gaussian(0.0, &[1.0], 0.5)];
    let model = model(3, &[(0, 1), (1, 2)], slots);
    let ctx = ExecutionContext::for_tests(9);
    let mut ws = MechanismWorkspace::default();
    let mut rng = CausalRng::from_seed(11);
    let evidence_nodes = [DenseNodeId::from_raw(2)];
    let evidence_values = [0.5];
    let evidence = ConditionEvidence { nodes: &evidence_nodes, values: &evidence_values };
    let bump =
        Intervention::transform(var(1), ValueExpr::generated().add(ValueExpr::constant(1.0)));
    let batch = sample_conditional_interventional(
        &model,
        std::slice::from_ref(&bump),
        &evidence,
        2_000,
        &mut rng,
        &mut ws,
        &ctx,
    )
    .unwrap();
    // y is clamped to the evidence and the transform is in force: x = (0.8 z + e) + 1.
    assert!(column(&batch, 2).iter().all(|v| *v == 0.5));
    let z = column(&batch, 0);
    let x = column(&batch, 1);
    let unexplained: f64 = x.iter().zip(&z).map(|(x, z)| x - 1.0 - 0.8 * z).map(f64::abs).sum();
    assert!(unexplained / 2_000.0 < 0.7, "x - 1 - 0.8 z is the structural residual");

    let on_x = [DenseNodeId::from_raw(1)];
    let evidence = ConditionEvidence { nodes: &on_x, values: &evidence_values };
    let error =
        sample_conditional_interventional(&model, &[bump], &evidence, 100, &mut rng, &mut ws, &ctx)
            .unwrap_err();
    assert!(error.to_string().contains("value program"), "{error}");
}

/// Standard normal distribution function (Abramowitz and Stegun 7.1.26, |error| < 1.5e-7).
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * (x.abs() / std::f64::consts::SQRT_2));
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-(x * x) / 2.0).exp();
    0.5 * (1.0 + erf.copysign(x))
}

fn normal_pdf(x: f64) -> f64 {
    (-(x * x) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

#[test]
fn downstream_means_under_a_price_floor_match_the_closed_form_expectation() {
    let (model, _) = pricing_chain();
    let n = 200_000;
    let floor =
        ValueExpr::constant(10.0).max(ValueExpr::constant(0.85).mul(ValueExpr::generated()));
    let base = observe(&model, n);
    let after = draw(&model, &[Intervention::transform(var(0), floor)], n);
    let mean = |batch: &ValueBatch, node: usize| column(batch, node).iter().sum::<f64>() / n as f64;

    // 0.85 P ~ N(9.35, 1.7^2) for P ~ N(11, 2^2); E max(c, Y) = c + (mu - c) Phi(d) + sigma phi(d)
    // with d = (mu - c) / sigma.
    let (mu, sigma, c) = (0.85 * 11.0, 0.85 * 2.0, 10.0);
    let d = (mu - c) / sigma;
    let floored_price = c + (mu - c) * normal_cdf(d) + sigma * normal_pdf(d);
    let demand = 50.0 - 0.8 * floored_price;
    let revenue = 3.0 + 0.5 * floored_price + 2.0 * demand;
    assert!((mean(&after, 0) - floored_price).abs() < 0.02, "price {}", mean(&after, 0));
    assert!((mean(&after, 1) - demand).abs() < 0.02, "demand {}", mean(&after, 1));
    assert!((mean(&after, 2) - revenue).abs() < 0.04, "revenue {}", mean(&after, 2));
    // The effect against the unintervened mean has the analytic sign and size: the floor
    // lowers the mean price from 11, so demand rises by 0.8 times the drop.
    let effect = mean(&after, 1) - mean(&base, 1);
    assert!(effect > 0.0 && (effect - 0.8 * (11.0 - floored_price)).abs() < 0.03, "{effect}");
}

#[test]
fn affine_and_comparison_programs_have_their_gaussian_closed_forms() {
    // x ~ N(0.5, 1), flag a leaf with no mechanism of its own.
    let slots = vec![gaussian(0.5, &[], 1.0), MechanismSlot::Constant { value: 0.0 }];
    let model = model(2, &[], slots);
    let n = 200_000;
    let affine = ValueExpr::constant(2.0).mul(ValueExpr::generated()).add(ValueExpr::constant(1.0));
    let flag = ValueExpr::var(var(0)).gt(ValueExpr::constant(1.5)).indicator();
    let batch = draw(
        &model,
        &[Intervention::transform(var(0), affine), Intervention::assign(var(1), flag)],
        n,
    );
    let x = column(&batch, 0);
    let mean = x.iter().sum::<f64>() / n as f64;
    let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    // 2 N(0.5, 1) + 1 = N(2, 4)
    assert!((mean - 2.0).abs() < 0.03 && (variance - 4.0).abs() < 0.08, "{mean} {variance}");
    // The flag reads the transformed x: P(N(2, 4) > 1.5) = Phi(0.25)
    let flag = column(&batch, 1);
    assert!(flag.iter().all(|v| *v == 0.0 || *v == 1.0));
    let rate = flag.iter().sum::<f64>() / n as f64;
    assert!((rate - normal_cdf(0.25)).abs() < 0.005, "{rate}");
    for (xi, fi) in x.iter().zip(&flag) {
        assert_eq!(*fi, f64::from(u8::from(*xi > 1.5)));
    }
}

#[test]
fn an_explicit_reference_reads_the_current_value_of_the_other_variable_in_the_same_draw() {
    // z(0) -> x(1): x := 0.5 z + x, with and without a transform on z itself.
    let slots = vec![gaussian(0.0, &[], 1.0), gaussian(0.0, &[0.7], 0.5)];
    let model = model(2, &[(0, 1)], slots);
    let n = 10_000;
    let base = observe(&model, n);
    let (z0, x0) = (column(&base, 0), column(&base, 1));
    let reads_z = ValueExpr::constant(0.5).mul(ValueExpr::var(var(0))).add(ValueExpr::generated());

    let plain = draw(&model, &[Intervention::transform(var(1), reads_z.clone())], n);
    let x = column(&plain, 1);
    for i in 0..n {
        assert_eq!(x[i], 0.5 * z0[i] + x0[i], "row {i}");
    }

    // When z is itself transformed, x reads the transformed z of the same draw, and its
    // mechanism sees it as a parent too.
    let shifted_z =
        Intervention::transform(var(0), ValueExpr::generated().add(ValueExpr::constant(2.0)));
    let both = draw(&model, &[shifted_z, Intervention::transform(var(1), reads_z)], n);
    let (z1, x1) = (column(&both, 0), column(&both, 1));
    for i in 0..n {
        assert_eq!(z1[i], z0[i] + 2.0);
        let generated = x0[i] + 0.7 * (z1[i] - z0[i]);
        assert!((x1[i] - (0.5 * z1[i] + generated)).abs() < 1e-9, "row {i}");
    }
}

#[test]
fn expression_faults_surface_from_intervention_validation() {
    let boolean =
        Intervention::transform(var(0), ValueExpr::generated().gt(ValueExpr::constant(0.0)));
    assert_eq!(
        boolean.validate(),
        Err(antecedent_core::InterventionError::Expression {
            fault: ExpressionFault::OutputType { found: antecedent_core::ExprType::Boolean }
        })
    );
}

type CallbackFn = dyn Fn(&[&[f64]], &mut [f64]) -> Result<(), CallbackError> + Send + Sync;

/// A Rust closure standing in for a caller-supplied batch function; counts its calls.
struct ClosureCallback {
    spec: CallbackSpec,
    calls: AtomicUsize,
    function: Box<CallbackFn>,
}

impl std::fmt::Debug for ClosureCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ClosureCallback({})", self.spec.name)
    }
}

impl BatchCallback for ClosureCallback {
    fn spec(&self) -> &CallbackSpec {
        &self.spec
    }

    fn call(&self, inputs: &[&[f64]], out: &mut [f64]) -> Result<(), CallbackError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        (self.function)(inputs, out)
    }
}

fn callback(
    name: &str,
    arity: usize,
    output: CallbackOutput,
    implementation: CallbackImplementation,
    function: impl Fn(&[&[f64]], &mut [f64]) -> Result<(), CallbackError> + Send + Sync + 'static,
) -> Arc<ClosureCallback> {
    Arc::new(ClosureCallback {
        spec: CallbackSpec {
            name: name.into(),
            inputs: (0..arity).map(|i| format!("x{i}")).collect::<Vec<_>>().into(),
            output,
            dependencies: Arc::from([PinnedDependency {
                name: "numpy".into(),
                version: "2.0.0".into(),
            }]),
            implementation,
        },
        calls: AtomicUsize::new(0),
        function: Box::new(function),
    })
}

fn runtime(
    name: &str,
    arity: usize,
    function: impl Fn(&[&[f64]], &mut [f64]) -> Result<(), CallbackError> + Send + Sync + 'static,
) -> Arc<ClosureCallback> {
    callback(name, arity, CallbackOutput::Real, CallbackImplementation::Runtime, function)
}

#[allow(clippy::unnecessary_wraps, reason = "the signature is the batch callback contract")]
fn price_floor(inputs: &[&[f64]], out: &mut [f64]) -> Result<(), CallbackError> {
    for (o, x) in out.iter_mut().zip(inputs[0]) {
        *o = 10.0_f64.max(0.85 * x);
    }
    Ok(())
}

#[test]
fn a_callback_floor_equals_the_expression_floor_on_every_node_with_one_call_per_batch() {
    let (model, _) = pricing_chain();
    let floor = runtime("floor", 1, price_floor);
    let by_callback = [Intervention::transform(
        var(0),
        ValueExpr::callback(floor.clone(), [ValueExpr::generated()]),
    )];
    let by_expression = [Intervention::transform(
        var(0),
        ValueExpr::constant(10.0).max(ValueExpr::constant(0.85).mul(ValueExpr::generated())),
    )];
    let n = 2_000;
    let (a, b) = (draw(&model, &by_callback, n), draw(&model, &by_expression, n));
    for node in 0..3 {
        let (ca, cb) = (column(&a, node), column(&b, node));
        assert!(ca.iter().zip(&cb).all(|(x, y)| x.to_bits() == y.to_bits()), "node {node}");
    }
    assert_eq!(floor.calls.load(Ordering::SeqCst), 1, "one call for the whole batch");
    // A second draw with the same seed is bit-identical (determinism across runs).
    let again = draw(&model, &by_callback, n);
    assert_eq!(column(&a, 2), column(&again, 2));
}

#[test]
fn callback_inputs_are_reads_like_any_expression_variable() {
    let (model, _) = pricing_chain();
    // demand := f(price) is an assignment from an ancestor: evaluated after price.
    let double = runtime("double", 1, |inputs, out| {
        for (o, x) in out.iter_mut().zip(inputs[0]) {
            *o = 2.0 * x;
        }
        Ok(())
    });
    let assign =
        Intervention::assign(var(1), ValueExpr::callback(double.clone(), [ValueExpr::var(var(0))]));
    let drawn = draw(&model, &[assign], 500);
    let (price, demand) = (column(&drawn, 0), column(&drawn, 1));
    assert!(price.iter().zip(&demand).all(|(p, d)| *d == 2.0 * p));

    // price := f(revenue) closes a cycle through the graph.
    let circular =
        Intervention::assign(var(0), ValueExpr::callback(double.clone(), [ValueExpr::var(var(2))]));
    expect_refusal(
        &model,
        &[circular],
        reason_code!("policy_dependency_circular"),
        "intervention_program.dependency_circular",
    );
    // A read of a variable the model does not have is unavailable.
    let unknown =
        Intervention::assign(var(1), ValueExpr::callback(double.clone(), [ValueExpr::var(var(9))]));
    expect_refusal(
        &model,
        &[unknown],
        reason_code!("policy_dependency_unavailable"),
        "intervention_program.variable_not_in_model",
    );
    // A history read has no window on a static model.
    let history = Intervention::assign(
        var(1),
        ValueExpr::callback(double.clone(), [ValueExpr::lag(var(0), 1)]),
    );
    expect_refusal(
        &model,
        &[history],
        reason_code!("policy_history_unavailable"),
        "intervention_program.history_unavailable",
    );
    // An assignment cannot hand its own generated value to the callback.
    let generated =
        Intervention::assign(var(1), ValueExpr::callback(double.clone(), [ValueExpr::generated()]));
    expect_refusal(
        &model,
        &[generated],
        reason_code!("policy_dependency_unavailable"),
        "intervention_program.generated_value_unavailable",
    );
    // Arguments must match the declared inputs.
    let arity = Intervention::assign(var(1), ValueExpr::callback(double, []));
    expect_refusal(
        &model,
        &[arity],
        reason_code!("invalid_argument"),
        "intervention_program.callback_declaration",
    );
}

#[test]
fn callback_outputs_are_validated_and_failures_keep_their_message() {
    let (model, _) = pricing_chain();
    let run = |callback: Arc<ClosureCallback>| {
        let iv = Intervention::transform(
            var(0),
            ValueExpr::callback(callback, [ValueExpr::generated()]),
        );
        let mut rng = CausalRng::from_seed(3);
        let mut ws = MechanismWorkspace::default();
        sample_interventional(&model, &[iv], 16, &mut rng, &mut ws, &ExecutionContext::for_tests(1))
    };
    let invalid = reason_code!("policy_callback_output_invalid");

    // Not finite: refused, never replaced by a default.
    let nan = runtime("nan_at_three", 1, |inputs, out| {
        out.copy_from_slice(inputs[0]);
        out[3] = f64::NAN;
        Ok(())
    });
    let error = run(nan).expect_err("NaN output");
    assert_eq!(
        refusal(&error),
        (invalid.to_string(), "intervention_program.callback_output".into())
    );
    assert!(
        error.to_string().contains("row 3") && error.to_string().contains("nan_at_three"),
        "{error}"
    );

    // A wrong length reported by the callback.
    let short = runtime("short", 1, |_, out| {
        Err(CallbackError::WrongLength { got: out.len() - 1, expected: out.len() })
    });
    let error = run(short).expect_err("short output");
    assert_eq!(refusal(&error).0, invalid);
    assert!(error.to_string().contains("expected exactly 16"), "{error}");

    // Declared support: levels and range.
    let levels = callback(
        "levels",
        1,
        CallbackOutput::levels([0.0, 1.0]),
        CallbackImplementation::Runtime,
        |_, out| {
            out.fill(0.0);
            out[5] = 2.0;
            Ok(())
        },
    );
    let error = run(levels).expect_err("outside the declared levels");
    assert_eq!(refusal(&error).0, invalid);
    assert!(error.to_string().contains("row 5"), "{error}");
    let ranged = callback(
        "ranged",
        1,
        CallbackOutput::Range { lower: 0.0, upper: 1.0 },
        CallbackImplementation::Runtime,
        |inputs, out| {
            out.copy_from_slice(inputs[0]);
            Ok(())
        },
    );
    assert_eq!(refusal(&run(ranged).expect_err("prices exceed [0, 1]")).0, invalid);

    // The callback's own failure surfaces with its message.
    let raising = runtime("raising", 1, |_, _| {
        Err(CallbackError::Raised { message: "ValueError: no demand model for region 7".into() })
    });
    let error = run(raising).expect_err("callback raised");
    assert_eq!(
        refusal(&error),
        (
            reason_code!("policy_callback_failed").to_string(),
            "intervention_program.callback_failed".into()
        )
    );
    assert!(error.to_string().contains("ValueError: no demand model for region 7"), "{error}");
}

#[test]
fn a_callback_is_licensed_for_simulation_and_conditionally_for_counterfactuals() {
    let (model, _) = pricing_chain();
    let floor = runtime("floor", 1, price_floor);
    let policy =
        Intervention::transform(var(0), ValueExpr::callback(floor, [ValueExpr::generated()]));
    let support = policy_support(&model, std::slice::from_ref(&policy));
    assert!(support.simulation.is_licensed());
    assert!(support.simulation.assumption().is_none());
    assert!(support.counterfactual.is_licensed());
    let assumption = support.counterfactual.assumption().expect("a conditional licence");
    assert!(
        assumption.contains("floor") && assumption.contains("declared inputs alone"),
        "{assumption}"
    );
    let antecedent_model::RouteDecision::Refused { code, .. } = &support.identification else {
        panic!("a callback has no observational identification");
    };
    assert_eq!(code, reason_code!("policy_route_not_licensed"));
}

#[test]
fn only_a_recoverable_callback_has_a_durable_form() {
    let digest = "ab".repeat(32);
    let recoverable = callback(
        "floor",
        1,
        CallbackOutput::Real,
        CallbackImplementation::Recoverable {
            source_sha256: digest.clone(),
            entry_point: "floor".into(),
        },
        price_floor,
    );
    let bound = runtime("closure", 1, price_floor);
    let with = |c: Arc<ClosureCallback>| {
        Intervention::transform(var(0), ValueExpr::callback(c, [ValueExpr::generated()]))
    };

    let closed_form = Intervention::transform(var(0), ValueExpr::generated().neg());
    assert_eq!(closed_form.portability(), Portability::Portable);

    let Portability::Recoverable(records) = with(recoverable.clone()).portability() else {
        panic!("a recoverable callback is exportable as a record");
    };
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].source_sha256, digest);
    // The consumer verifies the record against the implementation it supplies.
    records[0].verify(&recoverable.spec).unwrap();
    let tampered = callback(
        "floor",
        1,
        CallbackOutput::Real,
        CallbackImplementation::Recoverable {
            source_sha256: "cd".repeat(32),
            entry_point: "floor".into(),
        },
        price_floor,
    );
    assert_eq!(records[0].verify(&tampered.spec).unwrap_err().field, "source_sha256");
    assert!(records[0].verify(&bound.spec).is_err());

    // One runtime binding anywhere makes the whole program non-exportable.
    let sequence = Intervention::sequence(InterventionSequence::new(vec![
        SequencedIntervention::new(with(recoverable), TemporalPolicy::pulse(0)),
        SequencedIntervention::new(with(bound), TemporalPolicy::pulse(1)),
    ]));
    let portability = sequence.portability();
    assert_eq!(portability, Portability::RuntimeBound { callbacks: vec!["closure".into()] });
    let refusal = portability.export_refusal().expect("export refused");
    assert_eq!(refusal.code, reason_code!("policy_callback_not_portable"));
    assert!(refusal.message.contains("closure"), "{}", refusal.message);
}

//! Cross-world selective execution against enumerated, hand-computed truth.
//!
//! The response-type model: `X = U_X` with `P(U_X = 1) = 0.4`; `Y = f(X, U_Y)`
//! with `U_Y` over four response types (never, complier, defier, always) with
//! masses `q = [0.2, 0.5, 0.1, 0.2]`. Every expected value below is computed by
//! hand from `q` in the comment above it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::RequestedReading;
use antecedent_core::ExecutionContext;
use antecedent_counterfactual::selective::finite_scm::{MAX_UNITS, abduct_unit};
use antecedent_counterfactual::selective::{
    ExogenousInference, ExogenousVariable, FactualUnit, FiniteExecutionRequest, FiniteQuantity,
    FiniteScm, FiniteScmExecutor, FiniteVariable, FiniteWorld, NoiseCoupling, StageStatus,
    WorldAtom, WorldRoute,
};

const TOL: f64 = 1e-12;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

fn response_types(q: [f64; 4]) -> FiniteScm {
    FiniteScm {
        population: "pop-a".to_owned(),
        history: "regime-1".to_owned(),
        exogenous: vec![
            ExogenousVariable { name: "u_x".to_owned(), pmf: vec![0.6, 0.4] },
            ExogenousVariable { name: "u_y".to_owned(), pmf: q.to_vec() },
        ],
        variables: vec![
            FiniteVariable {
                name: "x".to_owned(),
                levels: 2,
                parents: vec![],
                exogenous: 0,
                table: vec![0, 1],
            },
            FiniteVariable {
                name: "y".to_owned(),
                levels: 2,
                parents: vec![0],
                exogenous: 1,
                // never, complier, defier, always; cfg = X.
                table: vec![0, 0, 0, 1, 1, 0, 1, 1],
            },
        ],
    }
}

fn do_x(name: &str, level: u8) -> FiniteWorld {
    FiniteWorld { name: name.to_owned(), interventions: vec![(0, level)], routes: vec![] }
}

fn unit(id: &str, observed: &[(usize, u8)]) -> FactualUnit {
    FactualUnit { unit: id.to_owned(), observed: observed.to_vec() }
}

fn request(
    scm: FiniteScm,
    units: Vec<FactualUnit>,
    worlds: Vec<FiniteWorld>,
    quantity: FiniteQuantity,
) -> FiniteExecutionRequest {
    FiniteExecutionRequest {
        evidence_history: scm.history.clone(),
        scm,
        units,
        worlds,
        quantity,
        coupling: NoiseCoupling::SharedAbducedExogenous,
        exogenous: ExogenousInference::ExactEnumeration,
        target_population: None,
        reading: RequestedReading::Point,
    }
}

fn atom(world: usize, variable: usize, level: u8) -> WorldAtom {
    WorldAtom { world, variable, level }
}

fn joint_y1_y0() -> FiniteQuantity {
    // Y_{x1} = 1 and Y_{x0} = 0 in worlds (0: do x0, 1: do x1).
    FiniteQuantity::JointProbability(vec![atom(1, 1, 1), atom(0, 1, 0)])
}

fn refusal_of(
    result: Result<
        impl std::fmt::Debug,
        antecedent_counterfactual::selective::CrossWorldExecutionError,
    >,
) -> (String, String) {
    let error = result.expect_err("a typed refusal");
    let refusal = error.refusal().expect("refusal");
    (refusal.code.to_owned(), refusal.detail.clone())
}

// P(Y_{x0}=0 | X=1, Y=1): the consistent response types are complier (0.5) and
// always (0.2); under do(x0) the complier gives 0 and always gives 1, so the
// probability of necessity is 0.5 / 0.7 = 5/7. P(evidence) = 0.4 * 0.7 = 0.28.
#[test]
fn probability_of_necessity_is_the_abduced_posterior_mass() {
    let req = request(
        response_types([0.2, 0.5, 0.1, 0.2]),
        vec![unit("case", &[(0, 1), (1, 1)])],
        vec![do_x("x0", 0), do_x("x1", 1)],
        FiniteQuantity::Probability(atom(0, 1, 0)),
    );
    let mut exec = FiniteScmExecutor::new();
    let out = exec.execute(&req, &ctx()).unwrap();
    assert!((out.value - 5.0 / 7.0).abs() < TOL, "{}", out.value);
    assert!((out.units[0].evidence_probability - 0.28).abs() < TOL);
    assert_eq!(out.label, "point_only");
    assert_eq!(out.work.mechanism_fits, 0);
    assert_eq!(out.work.shared_draws, 0);
    assert_eq!(out.work.abductions, 1);
    assert_eq!(out.work.abduction_states_enumerated, 8);
    assert_eq!(out.matrix_row, "counterfactual.finite_markovian_scm.marginal");
}

// No evidence: P(Y_{x1}=1, Y_{x0}=0) is the complier mass 0.5 under the shared
// exogenous term. The product of the marginals 0.7 * 0.7 = 0.49 is what
// independent per-world noise would give, and it is refused as a joint law.
#[test]
fn a_cross_world_joint_event_uses_the_coupled_worlds_only() {
    let base = request(
        response_types([0.2, 0.5, 0.1, 0.2]),
        vec![unit("u", &[])],
        vec![do_x("x0", 0), do_x("x1", 1)],
        joint_y1_y0(),
    );
    let mut exec = FiniteScmExecutor::new();
    let out = exec.execute(&base, &ctx()).unwrap();
    assert!((out.value - 0.5).abs() < TOL, "{}", out.value);
    assert!((out.value - 0.49).abs() > 1e-3);
    assert_eq!(out.matrix_row, "counterfactual.finite_markovian_scm.joint");

    let independent =
        FiniteExecutionRequest { coupling: NoiseCoupling::IndependentWorldNoise, ..base.clone() };
    let (code, detail) = refusal_of(exec.execute(&independent, &ctx()));
    assert_eq!(code, "joint_law_required");
    assert_eq!(detail, "joint_law_propagation.nonlinear_needs_aligned_joint_law");

    let undeclared = FiniteExecutionRequest { coupling: NoiseCoupling::Undeclared, ..base };
    let (code, detail) = refusal_of(exec.execute(&undeclared, &ctx()));
    assert_eq!(code, "joint_law_required");
    assert_eq!(detail, "joint_law_propagation.index_pairing_refused");
}

#[test]
fn rare_positive_factual_evidence_is_not_misreported_as_zero_probability() {
    let scm = FiniteScm {
        population: "pop-a".to_owned(),
        history: "regime-1".to_owned(),
        exogenous: vec![
            ExogenousVariable { name: "u_x".to_owned(), pmf: vec![1e-200, 1.0] },
            ExogenousVariable { name: "u_y".to_owned(), pmf: vec![1e-200, 1.0] },
        ],
        variables: vec![
            FiniteVariable {
                name: "x".to_owned(),
                levels: 2,
                parents: vec![],
                exogenous: 0,
                table: vec![1, 0],
            },
            FiniteVariable {
                name: "y".to_owned(),
                levels: 2,
                parents: vec![],
                exogenous: 1,
                table: vec![1, 0],
            },
        ],
    };
    let compiled = scm.compile().unwrap();
    let error = abduct_unit(&compiled, &unit("rare", &[(0, 1), (1, 1)]), &ctx())
        .expect_err("positive evidence below f64 range must fail closed");
    assert_eq!(
        error.refusal().unwrap().detail,
        "cross_world_execution.evidence_probability_underflow"
    );
}

#[test]
fn tolerated_exogenous_simplex_slack_is_normalized_for_abduction() {
    let scm = FiniteScm {
        population: "pop-a".to_owned(),
        history: "regime-1".to_owned(),
        exogenous: vec![ExogenousVariable {
            name: "u".to_owned(),
            pmf: vec![0.5, 0.500_000_000_5],
        }],
        variables: vec![FiniteVariable {
            name: "x".to_owned(),
            levels: 2,
            parents: vec![],
            exogenous: 0,
            table: vec![0, 1],
        }],
    };
    let source_digest = scm.digest();
    let compiled = scm.compile().unwrap();
    let posterior = abduct_unit(&compiled, &unit("observed", &[(0, 1)]), &ctx()).unwrap();
    assert!((posterior.evidence_probability - (0.500_000_000_5 / 1.000_000_000_5)).abs() < 1e-15);
    assert_eq!(posterior.states, vec![(1, 1.0)]);
    assert_eq!(scm.digest(), source_digest, "normalization does not rewrite source identity");
}

// Marginal means need no coupling: E[Y_{x1}] - E[Y_{x0}] = (0.5+0.2) - (0.1+0.2)
// = 0.4 for an undeclared coupling, but an interval reading is closed.
#[test]
fn marginal_contrasts_need_no_coupling_and_interval_readings_are_closed() {
    let base = FiniteExecutionRequest {
        coupling: NoiseCoupling::Undeclared,
        ..request(
            response_types([0.2, 0.5, 0.1, 0.2]),
            vec![unit("u", &[])],
            vec![do_x("x0", 0), do_x("x1", 1)],
            FiniteQuantity::MeanContrast { plus: (1, 1), minus: (0, 1) },
        )
    };
    let mut exec = FiniteScmExecutor::new();
    let out = exec.execute(&base, &ctx()).unwrap();
    assert!((out.value - 0.4).abs() < TOL, "{}", out.value);
    for reading in [RequestedReading::Interval, RequestedReading::Posterior] {
        let asked = FiniteExecutionRequest { reading, ..base.clone() };
        let (code, detail) = refusal_of(exec.execute(&asked, &ctx()));
        assert_eq!(code, "cell_not_licensed");
        assert!(detail.starts_with("joint_law_propagation."), "{detail}");
    }
}

// Mediation: X = U_X (0.5), M response types qm = [0.3, 0.4, 0.1, 0.2] of X,
// Y = M when U_Y = 0 (0.75) and Y = X when U_Y = 1 (0.25).
//   P(M_{x0}=1) = 0.1 + 0.2 = 0.3, P(M_{x1}=1) = 0.4 + 0.2 = 0.6
//   E[Y_{x1,M_x0}] = 0.75*0.3 + 0.25 = 0.475
//   E[Y_{x0,M_x0}] = 0.75*0.3        = 0.225
//   E[Y_{x0,M_x1}] = 0.75*0.6        = 0.45
//   E[Y_{x1,M_x1}] = 0.75*0.6 + 0.25 = 0.70
fn mediation() -> FiniteScm {
    FiniteScm {
        population: "pop-a".to_owned(),
        history: "regime-1".to_owned(),
        exogenous: vec![
            ExogenousVariable { name: "u_x".to_owned(), pmf: vec![0.5, 0.5] },
            ExogenousVariable { name: "u_m".to_owned(), pmf: vec![0.3, 0.4, 0.1, 0.2] },
            ExogenousVariable { name: "u_y".to_owned(), pmf: vec![0.75, 0.25] },
        ],
        variables: vec![
            FiniteVariable {
                name: "x".to_owned(),
                levels: 2,
                parents: vec![],
                exogenous: 0,
                table: vec![0, 1],
            },
            FiniteVariable {
                name: "m".to_owned(),
                levels: 2,
                parents: vec![0],
                exogenous: 1,
                table: vec![0, 0, 0, 1, 1, 0, 1, 1],
            },
            FiniteVariable {
                name: "y".to_owned(),
                levels: 2,
                parents: vec![0, 1],
                exogenous: 2,
                // cfg = 2 x + m. U_Y = 0: Y = m; U_Y = 1: Y = x.
                table: vec![0, 1, 0, 1, 0, 0, 1, 1],
            },
        ],
    }
}

fn mediation_worlds() -> Vec<FiniteWorld> {
    let route = |source| WorldRoute { parent: 1, child: 2, source };
    vec![
        // 0: do(x0), mediator-to-outcome read from world 1 (M_{x1}).
        FiniteWorld {
            name: "x0_m_of_x1".to_owned(),
            interventions: vec![(0, 0)],
            routes: vec![route(1)],
        },
        // 1: do(x1).
        do_x("x1", 1),
        // 2: do(x0).
        do_x("x0", 0),
        // 3: do(x1), mediator read from world 2 (M_{x0}).
        FiniteWorld {
            name: "x1_m_of_x0".to_owned(),
            interventions: vec![(0, 1)],
            routes: vec![route(2)],
        },
    ]
}

#[test]
fn natural_effects_are_recomputed_from_coupled_worlds() {
    let mut exec = FiniteScmExecutor::new();
    let run = |exec: &mut FiniteScmExecutor, quantity| {
        exec.execute(
            &request(mediation(), vec![unit("u", &[])], mediation_worlds(), quantity),
            &ctx(),
        )
        .unwrap()
    };
    // NDE = E[Y_{x1,M_x0}] - E[Y_{x0}] = 0.475 - 0.225 = 0.25.
    let nde = run(&mut exec, FiniteQuantity::MeanContrast { plus: (3, 2), minus: (2, 2) });
    assert!((nde.value - 0.25).abs() < TOL, "{}", nde.value);
    assert_eq!(nde.matrix_row, "natural_path_specific.finite_markovian_scm");
    // NIE = E[Y_{x0,M_x1}] - E[Y_{x0}] = 0.45 - 0.225 = 0.225.
    let nie = run(&mut exec, FiniteQuantity::MeanContrast { plus: (0, 2), minus: (2, 2) });
    assert!((nie.value - 0.225).abs() < TOL, "{}", nie.value);
    // Total effect E[Y_{x1}] - E[Y_{x0}] = 0.70 - 0.225 = 0.475 = NDE + the
    // reverse indirect effect E[Y_{x1}] - E[Y_{x1,M_x0}] = 0.70 - 0.475.
    let total = run(&mut exec, FiniteQuantity::MeanContrast { plus: (1, 2), minus: (2, 2) });
    assert!((total.value - 0.475).abs() < TOL);
    let reverse = run(&mut exec, FiniteQuantity::MeanContrast { plus: (1, 2), minus: (3, 2) });
    assert!((nde.value + reverse.value - total.value).abs() < TOL);
    // P(Y_{x1,M_x0} = 1 and Y_{x0} = 0): only U_Y = 1 separates the two worlds,
    // 0.25; independent worlds would give 0.475 * 0.775 = 0.368125.
    let joint =
        run(&mut exec, FiniteQuantity::JointProbability(vec![atom(3, 2, 1), atom(2, 2, 0)]));
    assert!((joint.value - 0.25).abs() < TOL, "{}", joint.value);
    assert!((joint.value - 0.368_125).abs() > 1e-2);
    // The four runs differ only by the quantity: one abduction in total.
    assert_eq!(exec.counters().abductions, 1);
    assert_eq!(exec.counters().abductions_reused, 4);
    assert_eq!(exec.counters().mechanism_fits, 0);
}

// Action-only reuse: the second request changes the worlds and the quantity
// only. Its abduction is reused (observed counters), its answer equals a fresh
// executor's bit for bit, and the stage receipts say so.
#[test]
fn an_action_only_change_reuses_abduction_and_matches_a_full_rerun_bit_for_bit() {
    let evidence = vec![unit("case", &[(0, 1), (1, 1)])];
    let first = request(
        response_types([0.2, 0.5, 0.1, 0.2]),
        evidence.clone(),
        vec![do_x("x0", 0), do_x("x1", 1)],
        FiniteQuantity::Probability(atom(0, 1, 0)),
    );
    let second = request(
        response_types([0.2, 0.5, 0.1, 0.2]),
        evidence,
        vec![do_x("x1", 1), do_x("x0", 0), do_x("x1b", 1)],
        FiniteQuantity::MeanContrast { plus: (0, 1), minus: (1, 1) },
    );
    let mut warm = FiniteScmExecutor::new();
    let a = warm.execute(&first, &ctx()).unwrap();
    assert_eq!(a.work.abductions, 1);
    let b = warm.execute(&second, &ctx()).unwrap();
    assert_eq!(b.work.abductions, 0, "abduction must not be redone");
    assert_eq!(b.work.abductions_reused, 1);
    assert_eq!(b.work.abduction_states_enumerated, 0);
    assert_eq!(b.work.world_predictions, 3);
    let status = |r: &antecedent_counterfactual::selective::FiniteExecutionResult, i: usize| {
        r.stages[i].status
    };
    assert_eq!(status(&b, 0), StageStatus::Declared);
    assert_eq!(status(&b, 1), StageStatus::Reused);
    assert_eq!(status(&b, 2), StageStatus::Reused);
    assert_eq!(status(&b, 3), StageStatus::Computed);
    // E[Y_{x1}|e] - E[Y_{x0}|e] = 1 - 2/7 = 5/7.
    assert!((b.value - 5.0 / 7.0).abs() < TOL);

    let mut fresh = FiniteScmExecutor::new();
    let c = fresh.execute(&second, &ctx()).unwrap();
    assert_eq!(c.work.abductions, 1);
    assert_eq!(b.value.to_bits(), c.value.to_bits());
    assert_eq!(b.units, c.units);
    assert_eq!(b.receipt, c.receipt);
    // The reused abduction carries the same identity as the fresh one.
    assert_eq!(a.receipt.abduction_digest, c.receipt.abduction_digest);
    assert_eq!(a.receipt.exogenous_digest, c.receipt.exogenous_digest);
    assert_ne!(a.receipt.prediction_digest, c.receipt.prediction_digest);
}

// Changed evidence, a changed mechanism parameter and a changed structure each
// rebuild the abduction and give the independently recomputed value.
#[test]
fn changed_evidence_or_mechanism_invalidates_abduction() {
    let q = [0.2, 0.5, 0.1, 0.2];
    let worlds = || vec![do_x("x0", 0), do_x("x1", 1)];
    let mut exec = FiniteScmExecutor::new();
    let base =
        request(response_types(q), vec![unit("case", &[(0, 1), (1, 1)])], worlds(), joint_y1_y0());
    let a = exec.execute(&base, &ctx()).unwrap();
    // Given X=1, Y=1: complier .5 and always .2; Y_{x1}=1 always, Y_{x0}=0 for
    // the complier only: 5/7.
    assert!((a.value - 5.0 / 7.0).abs() < TOL);

    // Evidence X=1, Y=0: never (.2) and defier (.1); Y_{x1}=1 needs ... the
    // defier gives Y_{x1}=0 and never gives 0, so the joint event has mass 0.
    let evidence =
        request(response_types(q), vec![unit("case", &[(0, 1), (1, 0)])], worlds(), joint_y1_y0());
    let b = exec.execute(&evidence, &ctx()).unwrap();
    assert_eq!(b.work.abductions, 1);
    assert_eq!(b.work.abductions_reused, 0);
    assert!(b.value.abs() < TOL);
    assert_ne!(a.receipt.abduction_digest, b.receipt.abduction_digest);

    // Mechanism parameter at unchanged shape: complier mass .4, defier .2.
    let params = request(
        response_types([0.2, 0.4, 0.2, 0.2]),
        vec![unit("case", &[])],
        worlds(),
        joint_y1_y0(),
    );
    let c = exec.execute(&params, &ctx()).unwrap();
    assert_eq!(c.work.abductions, 1);
    assert!((c.value - 0.4).abs() < TOL);
    let original = request(response_types(q), vec![unit("case", &[])], worlds(), joint_y1_y0());
    let d = exec.execute(&original, &ctx()).unwrap();
    assert_eq!(
        d.work.abductions, 1,
        "the evidence-free case was not retained for these parameters"
    );
    assert!((d.value - 0.5).abs() < TOL);
    assert_ne!(c.receipt.mechanism_digest, d.receipt.mechanism_digest);
    assert_ne!(c.receipt.abduction_digest, d.receipt.abduction_digest);

    // The retained identity is unchanged when the same request returns.
    let again = exec.execute(&original, &ctx()).unwrap();
    assert_eq!(again.work.abductions, 0);
    assert_eq!(again.value.to_bits(), d.value.to_bits());
}

#[test]
fn units_keep_their_identity_and_the_order_of_units_does_not_matter() {
    // Unit a has no evidence: P(Y_{x0}=0) = .2 + .5 = .7. Unit b has X=1,Y=1: 5/7.
    let make = |units| {
        request(
            response_types([0.2, 0.5, 0.1, 0.2]),
            units,
            vec![do_x("x0", 0), do_x("x1", 1)],
            FiniteQuantity::Probability(atom(0, 1, 0)),
        )
    };
    let mut e1 = FiniteScmExecutor::new();
    let one =
        e1.execute(&make(vec![unit("a", &[]), unit("b", &[(0, 1), (1, 1)])]), &ctx()).unwrap();
    let mut e2 = FiniteScmExecutor::new();
    let two =
        e2.execute(&make(vec![unit("b", &[(1, 1), (0, 1)]), unit("a", &[])]), &ctx()).unwrap();
    assert_eq!(one.units, two.units);
    assert_eq!(one.receipt, two.receipt);
    assert_eq!(one.value.to_bits(), two.value.to_bits());
    assert_eq!(one.units[0].unit, "a");
    assert!((one.units[0].value - 0.7).abs() < TOL);
    assert!((one.units[1].value - 5.0 / 7.0).abs() < TOL);
    assert!((one.value - (0.7 + 5.0 / 7.0) / 2.0).abs() < TOL);
}

#[test]
fn unsupported_requests_are_scientific_refusals_that_do_no_work() {
    let base = request(
        response_types([0.2, 0.5, 0.1, 0.2]),
        vec![unit("u", &[])],
        vec![do_x("x0", 0), do_x("x1", 1)],
        joint_y1_y0(),
    );
    let mut exec = FiniteScmExecutor::new();

    // A different target population: the transported row is closed and names
    // its prerequisites.
    let transported =
        FiniteExecutionRequest { target_population: Some("pop-b".to_owned()), ..base.clone() };
    let err = exec.execute(&transported, &ctx()).expect_err("closed");
    assert_eq!(err.refusal().unwrap().code, "cell_not_licensed");
    assert_eq!(err.refusal().unwrap().detail, "cross_world_execution.transported_row_closed");
    assert!(err.missing().iter().any(|m| m == "transport_license_missing"));
    assert!(err.missing().iter().any(|m| m == "target_population_abduction_receipt"));
    assert!(err.refusal().unwrap().remedy.is_some());

    // Evidence from another history needs a new compatibility check.
    let history =
        FiniteExecutionRequest { evidence_history: "regime-2".to_owned(), ..base.clone() };
    let (code, detail) = refusal_of(exec.execute(&history, &ctx()));
    assert_eq!(
        (code.as_str(), detail.as_str()),
        ("route_not_supported", "cross_world_execution.incompatible_history")
    );

    // A sampled posterior has no measured coverage record.
    let sampled =
        FiniteExecutionRequest { exogenous: ExogenousInference::PosteriorSampling, ..base.clone() };
    let (code, detail) = refusal_of(exec.execute(&sampled, &ctx()));
    assert_eq!(
        (code.as_str(), detail.as_str()),
        ("cell_not_licensed", "cross_world_execution.sampled_abduction_not_licensed")
    );

    // Refuted evidence: the complier-only model cannot produce X=1, Y=0.
    let refuted = request(
        response_types([0.0, 1.0, 0.0, 0.0]),
        vec![unit("u", &[(0, 1), (1, 0)])],
        vec![do_x("x0", 0)],
        FiniteQuantity::Probability(atom(0, 1, 0)),
    );
    let (code, detail) = refusal_of(exec.execute(&refuted, &ctx()));
    assert_eq!(
        (code.as_str(), detail.as_str()),
        ("route_not_supported", "cross_world_execution.evidence_outside_model_support")
    );

    // A route that is not an edge, and a quantity on an unknown world.
    let mut bad_route = base.clone();
    bad_route.worlds[1].routes = vec![WorldRoute { parent: 1, child: 0, source: 0 }];
    assert_eq!(
        refusal_of(exec.execute(&bad_route, &ctx())).1,
        "cross_world_execution.invalid_world"
    );
    let mut bad_quantity = base.clone();
    bad_quantity.quantity = FiniteQuantity::Probability(atom(5, 1, 0));
    assert_eq!(
        refusal_of(exec.execute(&bad_quantity, &ctx())).1,
        "cross_world_execution.invalid_quantity"
    );

    // Bounds.
    let mut many = base;
    many.units = (0..=MAX_UNITS).map(|i| unit(&format!("u{i}"), &[])).collect();
    assert_eq!(refusal_of(exec.execute(&many, &ctx())).1, "cross_world_execution.bounds_exceeded");

    // None of the refusals did abduction work or retained anything.
    assert_eq!(exec.counters().abductions, 0);
    assert_eq!(exec.retained_len(), 0);
}

#[test]
fn a_cancelled_execution_leaves_the_retained_state_unchanged() {
    let req = request(
        response_types([0.2, 0.5, 0.1, 0.2]),
        vec![unit("u", &[])],
        vec![do_x("x0", 0)],
        FiniteQuantity::Probability(atom(0, 1, 0)),
    );
    let mut exec = FiniteScmExecutor::new();
    exec.execute(&req, &ctx()).unwrap();
    let (retained, counters) = (exec.retained_len(), exec.counters());
    let cancelled = ExecutionContext::for_tests(3);
    cancelled.cancellation.cancel();
    let error = exec.execute(&req, &cancelled).expect_err("cancelled");
    assert_eq!(error, antecedent_counterfactual::selective::CrossWorldExecutionError::Cancelled);
    assert_eq!((exec.retained_len(), exec.counters()), (retained, counters));
}

// Shared exogenous term. U ~ (0.7, 0.3) feeds both X = U and Y = X xor U (so
// Y = 0 factually whenever the unit is observed at all). The model is fully
// declared, so the counterfactual is a computation, not an identification
// question.
fn shared_noise_scm() -> FiniteScm {
    FiniteScm {
        population: "pop-a".to_owned(),
        history: "regime-1".to_owned(),
        exogenous: vec![ExogenousVariable { name: "u".to_owned(), pmf: vec![0.7, 0.3] }],
        variables: vec![
            FiniteVariable {
                name: "x".to_owned(),
                levels: 2,
                parents: vec![],
                exogenous: 0,
                table: vec![0, 1],
            },
            FiniteVariable {
                name: "y".to_owned(),
                levels: 2,
                parents: vec![0],
                exogenous: 0,
                // [u * 2 + x]: u = 0 gives y = x, u = 1 gives y = 1 - x.
                table: vec![0, 1, 1, 0],
            },
        ],
    }
}

// The same mechanisms with an independent term per variable: only the law of
// U_y is shared with the model above, so factual X says nothing about it.
fn independent_noise_scm() -> FiniteScm {
    let mut scm = shared_noise_scm();
    scm.exogenous.push(ExogenousVariable { name: "u_y".to_owned(), pmf: vec![0.7, 0.3] });
    "u_x".clone_into(&mut scm.exogenous[0].name);
    scm.variables[1].exogenous = 1;
    scm
}

#[test]
fn a_shared_exogenous_term_is_executed_and_differs_from_independent_noise() {
    let worlds = vec![do_x("x0", 0), do_x("x1", 1)];
    let evidence = vec![unit("case", &[(0, 1)])];
    let atom_y1_x0 = FiniteQuantity::Probability(atom(0, 1, 1));

    // X = 1 forces U = 1 (posterior mass 1), so under do(x0) Y = 0 xor 1 = 1.
    let mut exec = FiniteScmExecutor::new();
    let shared = exec
        .execute(
            &request(shared_noise_scm(), evidence.clone(), worlds.clone(), atom_y1_x0.clone()),
            &ctx(),
        )
        .unwrap();
    assert_eq!(shared.matrix_row, "counterfactual.finite_shared_exogenous_scm.marginal");
    assert!((shared.value - 1.0).abs() < TOL);
    assert!((shared.units[0].evidence_probability - 0.3).abs() < TOL);

    // With a separate term for Y, factual X carries no information about it:
    // P(Y_x0 = 1) = P(U_y = 1) = 0.3.
    let independent = exec
        .execute(
            &request(independent_noise_scm(), evidence, worlds.clone(), atom_y1_x0.clone()),
            &ctx(),
        )
        .unwrap();
    assert_eq!(independent.matrix_row, "counterfactual.finite_markovian_scm.marginal");
    assert!((independent.value - 0.3).abs() < TOL);
    assert_ne!(shared.receipt.mechanism_digest, independent.receipt.mechanism_digest);
    assert_ne!(shared.receipt.abduction_digest, independent.receipt.abduction_digest);

    // Prior joint over the worlds, one realization of U in both: Y_x1 = 1 iff
    // U = 0 and Y_x0 = 1 iff U = 1, so the two never hold together, although
    // the marginals are 0.7 and 0.3 (an independent-noise product is 0.21).
    let joint = FiniteQuantity::JointProbability(vec![atom(1, 1, 1), atom(0, 1, 1)]);
    let coupled = exec
        .execute(&request(shared_noise_scm(), vec![unit("p", &[])], worlds, joint), &ctx())
        .unwrap();
    assert_eq!(coupled.matrix_row, "counterfactual.finite_shared_exogenous_scm.joint");
    assert!(coupled.value.abs() < TOL);
}

// Independent computation: structural equations written as plain functions and
// the exogenous states enumerated by hand, with no table, posterior or
// executor code. U (shared by M and Y) ~ (0.7, 0.3), E (X's own term) ~
// (0.5, 0.5); X = E, M = X or U, Y = X xor M xor U.
//
// `P(Y = 1 in the world do(X = x_do) with M read from the world do(X = m_do))`
// given the factual `(X, Y)`; `None` when the evidence has probability zero.
fn mediation_truth(evidence: Option<(u8, u8)>, x_do: u8, m_do: u8) -> Option<(f64, f64)> {
    let (mut mass, mut hit) = (0.0, 0.0);
    for (e, pe) in [(0u8, 0.5), (1, 0.5)] {
        for (u, pu) in [(0u8, 0.7), (1, 0.3)] {
            let factual_y = e ^ (e | u) ^ u;
            if evidence.is_some_and(|(ex, ey)| (e, factual_y) != (ex, ey)) {
                continue;
            }
            let w = pe * pu;
            mass += w;
            let mediator = m_do | u;
            hit += w * f64::from(x_do ^ mediator ^ u);
        }
    }
    (mass > 0.0).then_some((mass, hit / mass))
}

fn mediation_scm() -> FiniteScm {
    FiniteScm {
        population: "pop-a".to_owned(),
        history: "regime-1".to_owned(),
        exogenous: vec![
            ExogenousVariable { name: "e".to_owned(), pmf: vec![0.5, 0.5] },
            ExogenousVariable { name: "u".to_owned(), pmf: vec![0.7, 0.3] },
        ],
        variables: vec![
            FiniteVariable {
                name: "x".to_owned(),
                levels: 2,
                parents: vec![],
                exogenous: 0,
                table: vec![0, 1],
            },
            FiniteVariable {
                name: "m".to_owned(),
                levels: 2,
                parents: vec![0],
                exogenous: 1,
                // [u * 2 + x]: x or u.
                table: vec![0, 1, 1, 1],
            },
            FiniteVariable {
                name: "y".to_owned(),
                levels: 2,
                parents: vec![0, 1],
                exogenous: 1,
                // [u * 4 + x * 2 + m]: x xor m xor u.
                table: vec![0, 1, 1, 0, 1, 0, 0, 1],
            },
        ],
    }
}

fn nested_worlds(x_do: u8, m_do: u8) -> Vec<FiniteWorld> {
    // World 1 sets X = x_do and reads M from world 0, where X = m_do.
    let mut nested = do_x("nested", x_do);
    nested.routes = vec![WorldRoute { parent: 1, child: 2, source: 0 }];
    vec![do_x("mediator", m_do), nested]
}

#[test]
fn shared_exogenous_cross_world_values_equal_an_independent_enumeration() {
    for observed in [None, Some((1u8, 0u8)), Some((1, 1)), Some((0, 0)), Some((0, 1))] {
        let units = observed
            .map_or_else(|| vec![unit("u", &[])], |(x, y)| vec![unit("u", &[(0, x), (2, y)])]);
        let mut exec = FiniteScmExecutor::new();
        for (x_do, m_do) in [(1u8, 0u8), (0, 1), (0, 0), (1, 1)] {
            let out = exec.execute(
                &request(
                    mediation_scm(),
                    units.clone(),
                    nested_worlds(x_do, m_do),
                    FiniteQuantity::Probability(atom(1, 2, 1)),
                ),
                &ctx(),
            );
            match (out, mediation_truth(observed, x_do, m_do)) {
                (Ok(out), Some((mass, value))) => {
                    assert_eq!(out.matrix_row, "natural_path_specific.finite_shared_exogenous_scm");
                    assert!((out.value - value).abs() < TOL, "{observed:?} {x_do}{m_do}");
                    assert!((out.units[0].evidence_probability - mass).abs() < TOL);
                }
                (Err(error), None) => assert_eq!(
                    error.refusal().unwrap().detail,
                    "cross_world_execution.evidence_outside_model_support"
                ),
                (out, truth) => panic!("{observed:?}: {out:?} vs {truth:?}"),
            }
        }
    }
}

#[test]
fn a_shared_term_keeps_one_realization_across_the_factual_and_counterfactual_worlds() {
    // Y = 1 with X = 1 is only possible with U = 1 (probability 0.5 * 0.3),
    // and then do(x0) with M read from do(x1) gives Y = 0 ^ 1 ^ 1 = 0. With
    // Y = 0 and X = 1 the unit has U = 0 (probability 0.5 * 0.7) and the same
    // nested world gives Y = 1. Were U redrawn per world both would be 0.7.
    let mut exec = FiniteScmExecutor::new();
    let mut ask = |y: u8| {
        exec.execute(
            &request(
                mediation_scm(),
                vec![unit("u", &[(0, 1), (2, y)])],
                nested_worlds(0, 1),
                FiniteQuantity::Probability(atom(1, 2, 1)),
            ),
            &ctx(),
        )
        .unwrap()
    };
    let (high, low) = (ask(1), ask(0));
    assert!((high.units[0].evidence_probability - 0.15).abs() < TOL);
    assert!(high.value.abs() < TOL);
    assert!((low.units[0].evidence_probability - 0.35).abs() < TOL);
    assert!((low.value - 1.0).abs() < TOL);
}

#[test]
fn a_declared_but_oversized_joint_state_is_unsupported_not_unidentified() {
    // Six exogenous terms of eight states: 8^6 joint states, one variable each.
    let scm = FiniteScm {
        population: "pop-a".to_owned(),
        history: "regime-1".to_owned(),
        exogenous: (0..6)
            .map(|k| ExogenousVariable { name: format!("u{k}"), pmf: vec![0.125; 8] })
            .collect(),
        variables: (0..6)
            .map(|k| FiniteVariable {
                name: format!("v{k}"),
                levels: 2,
                parents: vec![],
                exogenous: k,
                table: vec![0, 1, 0, 1, 0, 1, 0, 1],
            })
            .collect(),
    };
    let req = request(
        scm,
        vec![unit("u", &[])],
        vec![do_x("w", 0)],
        FiniteQuantity::Probability(atom(0, 1, 1)),
    );
    let mut exec = FiniteScmExecutor::new();
    let (code, detail) = refusal_of(exec.execute(&req, &ctx()));
    assert_eq!(
        (code.as_str(), detail.as_str()),
        ("route_not_supported", "cross_world_execution.joint_state_bound_exceeded")
    );
    assert_eq!(exec.counters().abductions, 0);
}

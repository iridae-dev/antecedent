//! Linear programs with hand-derivable answers, degenerate and cycling-prone programs,
//! certificates rechecked by the verifier, vertex enumeration and conflicting constraint groups.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CancellationToken, ExecutionContext};
use antecedent_stats::linear_program::{
    ConstraintSense, InfeasibilityCertificate, LinearConstraint, LinearProgram, LpError, LpOptions,
    LpOutcome, LpSolution, irreducible_infeasible_groups,
    irreducible_infeasible_groups_with_context, maximize, smallest_infeasible_groups,
    smallest_infeasible_groups_with_context, solve, verify, vertices, vertices_with_context,
};

const INF: f64 = f64::INFINITY;

fn le(coefficients: &[f64], rhs: f64) -> LinearConstraint {
    LinearConstraint { coefficients: coefficients.to_vec(), sense: ConstraintSense::AtMost, rhs }
}

fn ge(coefficients: &[f64], rhs: f64) -> LinearConstraint {
    LinearConstraint { coefficients: coefficients.to_vec(), sense: ConstraintSense::AtLeast, rhs }
}

fn eq(coefficients: &[f64], rhs: f64) -> LinearConstraint {
    LinearConstraint { coefficients: coefficients.to_vec(), sense: ConstraintSense::Equal, rhs }
}

fn program(
    objective: &[f64],
    constraints: Vec<LinearConstraint>,
    lower: &[f64],
    upper: &[f64],
) -> LinearProgram {
    LinearProgram {
        objective: objective.to_vec(),
        constraints,
        lower: lower.to_vec(),
        upper: upper.to_vec(),
    }
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-9 * (1.0 + want.abs()), "{what}: got {got}, want {want}");
}

fn optimal(outcome: LpOutcome) -> LpSolution {
    match outcome {
        LpOutcome::Optimal(solution) => solution,
        other => panic!("expected an optimum, got {other:?}"),
    }
}

fn certificate(outcome: LpOutcome) -> InfeasibilityCertificate {
    match outcome {
        LpOutcome::Infeasible(certificate) => certificate,
        other => panic!("expected infeasible, got {other:?}"),
    }
}

fn options() -> LpOptions {
    LpOptions::default()
}

fn production() -> LinearProgram {
    // max 3x + 5y, x <= 4, 2y <= 12, 3x + 2y <= 18, x, y >= 0: optimum 36 at (2, 6).
    program(
        &[-3.0, -5.0],
        vec![le(&[1.0, 0.0], 4.0), le(&[0.0, 2.0], 12.0), le(&[3.0, 2.0], 18.0)],
        &[0.0, 0.0],
        &[INF, INF],
    )
}

#[test]
fn textbook_production_program_has_its_known_optimum_and_active_set() {
    let solution = optimal(solve(&production(), options()).unwrap());
    close(solution.objective, -36.0, "objective");
    close(solution.x[0], 2.0, "x");
    close(solution.x[1], 6.0, "y");
    assert_eq!(solution.active, vec![1, 2]);

    let mut positive = production();
    positive.objective = vec![3.0, 5.0];
    let maximum = optimal(maximize(&positive, options()).unwrap());
    close(maximum.objective, 36.0, "maximum");
}

#[test]
fn covering_program_with_at_least_constraints() {
    // min 2x + 3y + 4z, x + y >= 3, y + z >= 4, x + z >= 5: optimum 19 at (2, 1, 3).
    let p = program(
        &[2.0, 3.0, 4.0],
        vec![ge(&[1.0, 1.0, 0.0], 3.0), ge(&[0.0, 1.0, 1.0], 4.0), ge(&[1.0, 0.0, 1.0], 5.0)],
        &[0.0; 3],
        &[INF; 3],
    );
    let solution = optimal(solve(&p, options()).unwrap());
    close(solution.objective, 19.0, "objective");
    for (got, want) in solution.x.iter().zip([2.0, 1.0, 3.0]) {
        close(*got, want, "x");
    }
}

#[test]
fn beale_cycling_example_terminates_at_its_optimum() {
    let p = program(
        &[-0.75, 150.0, -0.02, 6.0],
        vec![
            le(&[0.25, -60.0, -0.04, 9.0], 0.0),
            le(&[0.5, -90.0, -0.02, 3.0], 0.0),
            le(&[0.0, 0.0, 1.0, 0.0], 1.0),
        ],
        &[0.0; 4],
        &[INF; 4],
    );
    let solution = optimal(solve(&p, options()).unwrap());
    close(solution.objective, -0.05, "objective");
    for (got, want) in solution.x.iter().zip([0.04, 0.0, 1.0, 0.0]) {
        close(*got, want, "x");
    }
}

#[test]
fn highly_degenerate_vertex_with_duplicate_rows() {
    // Seven constraints pass through the optimum (1, 1, 1), several of them duplicates.
    let p = program(
        &[-1.0, -1.0, -1.0],
        vec![
            le(&[1.0, 0.0, 0.0], 1.0),
            le(&[1.0, 0.0, 0.0], 1.0),
            le(&[0.0, 1.0, 0.0], 1.0),
            le(&[0.0, 0.0, 1.0], 1.0),
            le(&[1.0, 1.0, 0.0], 2.0),
            le(&[1.0, 1.0, 1.0], 3.0),
            le(&[2.0, 2.0, 2.0], 6.0),
        ],
        &[0.0; 3],
        &[INF; 3],
    );
    let solution = optimal(solve(&p, options()).unwrap());
    close(solution.objective, -3.0, "objective");
    assert_eq!(solution.active, (0..7).collect::<Vec<_>>());
}

#[test]
fn equality_constraints_and_free_variables() {
    // x + y = 1, x, y >= 0: min x - y = -1 at (0, 1).
    let p = program(&[1.0, -1.0], vec![eq(&[1.0, 1.0], 1.0)], &[0.0, 0.0], &[INF, INF]);
    let solution = optimal(solve(&p, options()).unwrap());
    close(solution.objective, -1.0, "equality");
    // Free x, y with x + y >= 2 and x - y = 0: min x + y = 2 at (1, 1).
    let p = program(
        &[1.0, 1.0],
        vec![ge(&[1.0, 1.0], 2.0), eq(&[1.0, -1.0], 0.0)],
        &[-INF, -INF],
        &[INF, INF],
    );
    let solution = optimal(solve(&p, options()).unwrap());
    close(solution.objective, 2.0, "free");
    close(solution.x[0], 1.0, "x");
    close(solution.x[1], 1.0, "y");
    // A free variable the objective ignores keeps a valid optimum.
    let p = program(&[1.0, 0.0], vec![ge(&[1.0, 0.0], -3.0)], &[-INF, -INF], &[INF, INF]);
    close(optimal(solve(&p, options()).unwrap()).objective, -3.0, "ignored free variable");
}

#[test]
fn fixed_bounds_empty_constraints_and_zero_variables() {
    let p = program(&[3.0, -2.0], vec![le(&[1.0, 1.0], 10.0)], &[1.0, 2.0], &[1.0, 2.0]);
    let solution = optimal(solve(&p, options()).unwrap());
    close(solution.objective, -1.0, "fixed");
    let p = program(&[1.0, 1.0], vec![ge(&[1.0, 1.0], 5.0)], &[1.0, 1.0], &[1.0, 1.0]);
    let c = certificate(solve(&p, options()).unwrap());
    verify(&p, &LpOutcome::Infeasible(c), options()).unwrap();

    let p = program(&[1.0, -1.0], Vec::new(), &[-1.0, -2.0], &[1.0, 2.0]);
    close(optimal(solve(&p, options()).unwrap()).objective, -3.0, "bounds only");

    let empty = program(&[], Vec::new(), &[], &[]);
    let solution = optimal(solve(&empty, options()).unwrap());
    assert!(solution.x.is_empty() && solution.active.is_empty());
    close(solution.objective, 0.0, "zero variables");
    let satisfied = program(&[], vec![le(&[], 1.0), eq(&[], 0.0)], &[], &[]);
    assert_eq!(optimal(solve(&satisfied, options()).unwrap()).active, vec![1]);
    let violated = program(&[], vec![le(&[], 1.0), ge(&[], 2.0)], &[], &[]);
    let c = certificate(solve(&violated, options()).unwrap());
    assert!(c.constraint_multipliers[1] < 0.0);
}

#[test]
fn infeasible_programs_carry_certificates_that_recheck() {
    let cases = vec![
        program(
            &[1.0, 0.0],
            vec![ge(&[1.0, 0.0], 3.0), le(&[1.0, 0.0], 2.0)],
            &[-INF, 0.0],
            &[INF, 1.0],
        ),
        program(
            &[0.0; 3],
            vec![eq(&[1.0, 1.0, 0.0], 1.0), eq(&[0.0, 1.0, 1.0], 1.0), eq(&[1.0, 2.0, 1.0], 3.0)],
            &[-5.0; 3],
            &[5.0; 3],
        ),
        program(&[1.0, 1.0], vec![ge(&[1.0, 1.0], 3.0)], &[0.0, 0.0], &[1.0, 1.0]),
        program(&[1.0], vec![ge(&[2.0], 7.0), le(&[1.0], 1.0)], &[-INF], &[INF]),
    ];
    for p in cases {
        let c = certificate(solve(&p, options()).unwrap());
        verify(&p, &LpOutcome::Infeasible(c.clone()), options()).unwrap();
        // Every constraint and bound sign rule holds.
        for (constraint, y) in p.constraints.iter().zip(&c.constraint_multipliers) {
            match constraint.sense {
                ConstraintSense::AtMost => assert!(*y >= 0.0),
                ConstraintSense::AtLeast => assert!(*y <= 0.0),
                ConstraintSense::Equal => {}
            }
        }
        assert!(c.lower_multipliers.iter().chain(&c.upper_multipliers).all(|v| *v >= 0.0));
    }
}

#[test]
fn tampered_certificates_fail_verification() {
    let p = program(&[1.0], vec![ge(&[1.0], 3.0), le(&[1.0], 2.0)], &[-INF], &[INF]);
    let c = certificate(solve(&p, options()).unwrap());
    let mut flipped = c.clone();
    flipped.constraint_multipliers[1] = -flipped.constraint_multipliers[1];
    assert_eq!(
        verify(&p, &LpOutcome::Infeasible(flipped), options()),
        Err(LpError::VerificationFailed { check: "multiplier_sign" })
    );
    let zero = InfeasibilityCertificate {
        constraint_multipliers: vec![0.0, 0.0],
        lower_multipliers: vec![0.0],
        upper_multipliers: vec![0.0],
    };
    assert_eq!(
        verify(&p, &LpOutcome::Infeasible(zero), options()),
        Err(LpError::VerificationFailed { check: "empty_certificate" })
    );
    let unbalanced = InfeasibilityCertificate {
        constraint_multipliers: vec![-1.0, 2.0],
        lower_multipliers: vec![0.0],
        upper_multipliers: vec![0.0],
    };
    assert_eq!(
        verify(&p, &LpOutcome::Infeasible(unbalanced), options()),
        Err(LpError::VerificationFailed { check: "combination_not_zero" })
    );
    let on_infinite = InfeasibilityCertificate {
        constraint_multipliers: c.constraint_multipliers,
        lower_multipliers: vec![1.0],
        upper_multipliers: vec![0.0],
    };
    assert_eq!(
        verify(&p, &LpOutcome::Infeasible(on_infinite), options()),
        Err(LpError::VerificationFailed { check: "multiplier_on_infinite_bound" })
    );
    // A feasible program has no contradiction to certify.
    let feasible = program(&[1.0], vec![le(&[1.0], 2.0)], &[0.0], &[INF]);
    let claim = InfeasibilityCertificate {
        constraint_multipliers: vec![1.0],
        lower_multipliers: vec![1.0],
        upper_multipliers: vec![0.0],
    };
    assert_eq!(
        verify(&feasible, &LpOutcome::Infeasible(claim), options()),
        Err(LpError::VerificationFailed { check: "no_contradiction" })
    );
}

#[test]
fn unbounded_programs_return_a_feasible_improving_ray() {
    let cases = vec![
        program(&[-1.0, 0.0], vec![le(&[1.0, -1.0], 1.0)], &[0.0, 0.0], &[INF, INF]),
        program(&[1.0], Vec::new(), &[-INF], &[INF]),
        program(&[1.0, 1.0], vec![eq(&[1.0, -1.0], 0.0)], &[-INF, -INF], &[0.0, INF]),
    ];
    for p in cases {
        let outcome = solve(&p, options()).unwrap();
        let LpOutcome::Unbounded { ref point, ref ray } = outcome else {
            panic!("expected unbounded, got {outcome:?}");
        };
        let decrease: f64 = p.objective.iter().zip(ray).map(|(c, d)| c * d).sum();
        assert!(decrease < 0.0);
        assert_eq!(point.len(), p.objective.len());
        verify(&p, &outcome, options()).unwrap();
        let reversed =
            LpOutcome::Unbounded { point: point.clone(), ray: ray.iter().map(|v| -v).collect() };
        assert!(verify(&p, &reversed, options()).is_err());
    }
}

#[test]
fn verifier_rejects_wrong_optima() {
    let p = production();
    let good = optimal(solve(&p, options()).unwrap());
    let suboptimal = LpSolution { x: vec![0.0, 0.0], objective: 0.0, active: Vec::new() };
    assert_eq!(
        verify(&p, &LpOutcome::Optimal(suboptimal), options()),
        Err(LpError::VerificationFailed { check: "optimality_not_proven" })
    );
    let wrong_value = LpSolution { objective: -35.0, ..good.clone() };
    assert_eq!(
        verify(&p, &LpOutcome::Optimal(wrong_value), options()),
        Err(LpError::VerificationFailed { check: "objective_mismatch" })
    );
    let outside = LpSolution { x: vec![3.0, 6.0], objective: -39.0, active: Vec::new() };
    assert_eq!(
        verify(&p, &LpOutcome::Optimal(outside), options()),
        Err(LpError::VerificationFailed { check: "constraint_violated" })
    );
    let loose_active = LpSolution { active: vec![0, 1, 2], ..good.clone() };
    assert_eq!(
        verify(&p, &LpOutcome::Optimal(loose_active), options()),
        Err(LpError::VerificationFailed { check: "active_not_tight" })
    );
    let short = LpSolution { x: vec![2.0], ..good };
    assert!(matches!(
        verify(&p, &LpOutcome::Optimal(short), options()),
        Err(LpError::OutcomeLength { part: "x", expected: 2, found: 1 })
    ));
}

#[test]
fn optimum_does_not_depend_on_constraint_order() {
    let base = vec![
        le(&[1.0, 2.0, 1.0], 8.0),
        ge(&[1.0, -1.0, 0.0], -2.0),
        le(&[3.0, 1.0, 2.0], 12.0),
        eq(&[1.0, 1.0, 1.0], 5.0),
        ge(&[0.0, 1.0, 2.0], 2.0),
    ];
    let objective = [-2.0, -3.0, 1.0];
    let reference = optimal(
        solve(&program(&objective, base.clone(), &[0.0; 3], &[4.0; 3]), options()).unwrap(),
    )
    .objective;
    let orders = [[4, 3, 2, 1, 0], [2, 0, 4, 1, 3], [1, 4, 0, 3, 2]];
    for order in orders {
        let permuted: Vec<_> = order.iter().map(|&i| base[i].clone()).collect();
        let value = optimal(
            solve(&program(&objective, permuted, &[0.0; 3], &[4.0; 3]), options()).unwrap(),
        )
        .objective;
        close(value, reference, "permuted objective");
    }
}

#[test]
fn larger_random_program_is_solved_and_verified() {
    // 40 variables, 600 constraints placed around a known interior point.
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut uniform = move || {
        state =
            state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1_u64 << 53) as f64)
    };
    let n = 40;
    let anchor: Vec<f64> = (0..n).map(|_| 4.0 * uniform() - 2.0).collect();
    let mut constraints = Vec::new();
    for _ in 0..600 {
        let a: Vec<f64> = (0..n).map(|_| 2.0 * uniform() - 1.0).collect();
        let at_anchor: f64 = a.iter().zip(&anchor).map(|(x, y)| x * y).sum();
        constraints.push(le(&a, at_anchor + uniform()));
    }
    let objective: Vec<f64> = (0..n).map(|_| 2.0 * uniform() - 1.0).collect();
    let p = program(&objective, constraints, &vec![-10.0; n], &vec![10.0; n]);
    let solution = optimal(solve(&p, options()).unwrap());
    let at_anchor: f64 = objective.iter().zip(&anchor).map(|(x, y)| x * y).sum();
    assert!(solution.objective <= at_anchor);
}

#[test]
fn vertices_of_cube_simplex_cut_cube_and_degenerate_pyramid() {
    let cube = vertices(&[], &[0.0; 3], &[1.0; 3], 100, options()).unwrap();
    assert_eq!(cube.len(), 8);
    assert_eq!(cube[0], vec![0.0, 0.0, 0.0]);
    assert_eq!(cube[7], vec![1.0, 1.0, 1.0]);

    let simplex =
        vertices(&[le(&[1.0, 1.0, 1.0], 1.0)], &[0.0; 3], &[INF; 3], 100, options()).unwrap();
    let expected = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    assert_eq!(simplex.len(), 4);
    for (got, want) in simplex.iter().zip(expected) {
        for (g, w) in got.iter().zip(want) {
            close(*g, w, "simplex vertex");
        }
    }

    let cut = vertices(&[le(&[1.0, 1.0, 1.0], 2.0)], &[0.0; 3], &[1.0; 3], 100, options()).unwrap();
    assert_eq!(cut.len(), 7);
    assert!(cut.iter().all(|v| v.iter().sum::<f64>() <= 2.0 + 1e-12));

    // Square pyramid with apex (0, 0, 1): four facets meet at the apex, plus redundant rows.
    let pyramid = vec![
        le(&[1.0, 0.0, 1.0], 1.0),
        le(&[-1.0, 0.0, 1.0], 1.0),
        le(&[0.0, 1.0, 1.0], 1.0),
        le(&[0.0, -1.0, 1.0], 1.0),
        le(&[1.0, 1.0, 2.0], 2.0),
        le(&[1.0, 1.0, 1.0], 3.0),
        le(&[2.0, 0.0, 2.0], 2.0),
    ];
    let points =
        vertices(&pyramid, &[-1.0, -1.0, 0.0], &[1.0, 1.0, INF], 10_000, options()).unwrap();
    assert_eq!(points.len(), 5);
    let apex = points.iter().filter(|v| (v[2] - 1.0).abs() < 1e-12).count();
    assert_eq!(apex, 1);
    for w in points.windows(2) {
        assert!(
            w[0].iter()
                .zip(&w[1])
                .map(|(a, b)| a.total_cmp(b))
                .find(|o| o.is_ne())
                .is_some_and(std::cmp::Ordering::is_lt)
        );
    }
}

#[test]
fn vertex_enumeration_of_empty_unbounded_and_oversized_regions() {
    let empty = vertices(&[ge(&[1.0, 1.0], 3.0)], &[0.0; 2], &[1.0; 2], 100, options()).unwrap();
    assert!(empty.is_empty());
    assert_eq!(
        vertices(&[le(&[1.0, -1.0], 0.0)], &[0.0; 2], &[INF; 2], 100, options()),
        Err(LpError::UnboundedRegion { variable: 0 })
    );
    // Cube: 6 bound hyperplanes, C(6, 3) = 20 candidate bases.
    assert_eq!(
        vertices(&[], &[0.0; 3], &[1.0; 3], 19, options()),
        Err(LpError::VertexCapExceeded { candidate_bases: 20, cap: 19 })
    );
    let point = vertices(&[], &[], &[], 1, options()).unwrap();
    assert_eq!(point, vec![Vec::<f64>::new()]);
}

fn group_example() -> (Vec<LinearConstraint>, Vec<usize>) {
    // Group 10: x >= 5. Group 20: x <= 1. Group 30: y <= 1 (two copies). Group 40: x + y >= 4.
    // Conflicts: {10, 20} and {20, 30, 40}.
    let constraints = vec![
        ge(&[1.0, 0.0], 5.0),
        le(&[1.0, 0.0], 1.0),
        le(&[0.0, 1.0], 1.0),
        le(&[0.0, 2.0], 2.0),
        ge(&[1.0, 1.0], 4.0),
    ];
    (constraints, vec![10, 20, 30, 30, 40])
}

#[test]
fn deletion_filter_and_smallest_conflict() {
    let (constraints, group) = group_example();
    let bounds = ([0.0, 0.0], [10.0, 10.0]);
    // Deletion in ascending order drops 10 first, leaving the three-group conflict.
    let irreducible =
        irreducible_infeasible_groups(&constraints, &group, &bounds.0, &bounds.1, options())
            .unwrap();
    assert_eq!(irreducible, Some(vec![20, 30, 40]));
    let smallest =
        smallest_infeasible_groups(&constraints, &group, &bounds.0, &bounds.1, 3, options())
            .unwrap();
    assert_eq!(smallest, Some(vec![10, 20]));
    let none_small =
        smallest_infeasible_groups(&constraints, &group, &bounds.0, &bounds.1, 1, options())
            .unwrap();
    assert_eq!(none_small, None);

    // Declaration order of the constraints does not change either answer.
    let order = [4, 2, 0, 3, 1];
    let permuted: Vec<_> = order.iter().map(|&i| constraints[i].clone()).collect();
    let permuted_group: Vec<_> = order.iter().map(|&i| group[i]).collect();
    assert_eq!(
        irreducible_infeasible_groups(&permuted, &permuted_group, &bounds.0, &bounds.1, options())
            .unwrap(),
        Some(vec![20, 30, 40])
    );
    assert_eq!(
        smallest_infeasible_groups(&permuted, &permuted_group, &bounds.0, &bounds.1, 4, options())
            .unwrap(),
        Some(vec![10, 20])
    );

    // A bound can be one side of the conflict.
    let single = vec![ge(&[1.0, 0.0], 11.0), le(&[0.0, 1.0], 3.0)];
    assert_eq!(
        smallest_infeasible_groups(&single, &[0, 1], &bounds.0, &bounds.1, 2, options()).unwrap(),
        Some(vec![0])
    );
    // Feasible constraints have no conflict.
    let feasible = vec![le(&[1.0, 1.0], 3.0)];
    assert_eq!(
        irreducible_infeasible_groups(&feasible, &[0], &bounds.0, &bounds.1, options()).unwrap(),
        None
    );
    assert_eq!(
        smallest_infeasible_groups(&feasible, &[0], &bounds.0, &bounds.1, 1, options()).unwrap(),
        None
    );
}

#[test]
fn every_refusal_has_its_own_variant() {
    let p = production();
    let mut bad = p.clone();
    bad.lower = vec![0.0];
    assert!(matches!(solve(&bad, options()), Err(LpError::BoundsLength { .. })));
    let mut bad = p.clone();
    bad.constraints[1].coefficients = vec![1.0];
    assert!(matches!(
        solve(&bad, options()),
        Err(LpError::ConstraintLength { constraint: 1, variables: 2, found: 1 })
    ));
    let mut bad = p.clone();
    bad.objective[1] = f64::NAN;
    assert_eq!(solve(&bad, options()), Err(LpError::NonFiniteObjective { variable: 1 }));
    let mut bad = p.clone();
    bad.constraints[2].coefficients[0] = INF;
    assert_eq!(
        solve(&bad, options()),
        Err(LpError::NonFiniteCoefficient { constraint: 2, variable: 0 })
    );
    let mut bad = p.clone();
    bad.constraints[0].rhs = f64::NAN;
    assert_eq!(solve(&bad, options()), Err(LpError::NonFiniteRhs { constraint: 0 }));
    let mut bad = p.clone();
    bad.upper[1] = f64::NAN;
    assert_eq!(solve(&bad, options()), Err(LpError::NanBound { variable: 1 }));
    let mut bad = p.clone();
    bad.lower[0] = INF;
    assert_eq!(solve(&bad, options()), Err(LpError::InfiniteBoundWrongSide { variable: 0 }));
    let mut bad = p.clone();
    bad.lower[0] = 2.0;
    bad.upper[0] = 1.0;
    assert!(matches!(solve(&bad, options()), Err(LpError::InvertedBounds { variable: 0, .. })));
    let loose = LpOptions { feasibility_tolerance: 0.0, ..options() };
    assert_eq!(solve(&p, loose), Err(LpError::InvalidOptions { option: "feasibility_tolerance" }));
    let loose = LpOptions { optimality_tolerance: f64::NAN, ..options() };
    assert_eq!(solve(&p, loose), Err(LpError::InvalidOptions { option: "optimality_tolerance" }));
    let tight = LpOptions { max_pivots: 0, ..options() };
    assert_eq!(solve(&p, tight), Err(LpError::PivotLimit { max_pivots: 0 }));
    let (constraints, _) = group_example();
    assert_eq!(
        irreducible_infeasible_groups(&constraints, &[0, 1], &[0.0; 2], &[1.0; 2], options()),
        Err(LpError::GroupLength { constraints: 5, groups: 2 })
    );
    let wrong_shape = LpOutcome::Infeasible(InfeasibilityCertificate {
        constraint_multipliers: vec![1.0],
        lower_multipliers: vec![0.0; 2],
        upper_multipliers: vec![0.0; 2],
    });
    assert!(matches!(
        verify(&p, &wrong_shape, options()),
        Err(LpError::OutcomeLength { part: "constraint_multipliers", .. })
    ));
    assert!(matches!(
        vertices(&[le(&[1.0], 1.0)], &[0.0; 2], &[1.0; 2], 10, options()),
        Err(LpError::ConstraintLength { .. })
    ));
}

#[test]
fn detail_ids_are_distinct_and_namespaced() {
    let errors = [
        LpError::BoundsLength { variables: 1, lower: 0, upper: 1 },
        LpError::ConstraintLength { constraint: 0, variables: 1, found: 2 },
        LpError::GroupLength { constraints: 1, groups: 0 },
        LpError::OutcomeLength { part: "x", expected: 1, found: 0 },
        LpError::NonFiniteObjective { variable: 0 },
        LpError::NonFiniteCoefficient { constraint: 0, variable: 0 },
        LpError::NonFiniteRhs { constraint: 0 },
        LpError::NanBound { variable: 0 },
        LpError::InfiniteBoundWrongSide { variable: 0 },
        LpError::InvertedBounds { variable: 0, lower: 1.0, upper: 0.0 },
        LpError::InvalidOptions { option: "feasibility_tolerance" },
        LpError::PivotLimit { max_pivots: 0 },
        LpError::NumericalBreakdown { stage: "working_matrix" },
        LpError::VerificationFailed { check: "constraint_violated" },
        LpError::UnboundedRegion { variable: 0 },
        LpError::VertexCapExceeded { candidate_bases: 2, cap: 1 },
    ];
    let mut ids: Vec<&str> = errors.iter().map(LpError::detail).collect();
    assert!(ids.iter().all(|id| id.starts_with("linear_program.")));
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), errors.len());
}

#[test]
fn cancellation_stops_vertex_enumeration_and_the_conflict_searches() {
    let (constraints, group) = group_example();
    let bounds = ([0.0, 0.0], [10.0, 10.0]);
    let mut ctx = ExecutionContext::production(0, 1);
    ctx.cancellation.cancel();
    assert_eq!(
        vertices_with_context(&[], &[0.0; 3], &[1.0; 3], 100, options(), &ctx),
        Err(LpError::Cancelled)
    );
    assert_eq!(
        irreducible_infeasible_groups_with_context(
            &constraints,
            &group,
            &bounds.0,
            &bounds.1,
            options(),
            &ctx
        ),
        Err(LpError::Cancelled)
    );
    assert_eq!(
        smallest_infeasible_groups_with_context(
            &constraints,
            &group,
            &bounds.0,
            &bounds.1,
            3,
            options(),
            &ctx
        ),
        Err(LpError::Cancelled)
    );
    // Mid-run: the cube has 20 candidate bases and the smallest-conflict search solves more
    // than three feasibility programs before it finds {10, 20}.
    ctx.cancellation = CancellationToken::cancel_after_checks(5);
    assert_eq!(
        vertices_with_context(&[], &[0.0; 3], &[1.0; 3], 100, options(), &ctx),
        Err(LpError::Cancelled)
    );
    ctx.cancellation = CancellationToken::cancel_after_checks(3);
    assert_eq!(
        smallest_infeasible_groups_with_context(
            &constraints,
            &group,
            &bounds.0,
            &bounds.1,
            3,
            options(),
            &ctx
        ),
        Err(LpError::Cancelled)
    );
    // An uncancelled context gives the context-free answers.
    ctx.cancellation = CancellationToken::new();
    assert_eq!(
        vertices_with_context(&[], &[0.0; 3], &[1.0; 3], 100, options(), &ctx).unwrap(),
        vertices(&[], &[0.0; 3], &[1.0; 3], 100, options()).unwrap()
    );
    assert_eq!(
        smallest_infeasible_groups_with_context(
            &constraints,
            &group,
            &bounds.0,
            &bounds.1,
            3,
            options(),
            &ctx
        )
        .unwrap(),
        Some(vec![10, 20])
    );
    assert_eq!(LpError::Cancelled.detail(), "linear_program.cancelled");
}

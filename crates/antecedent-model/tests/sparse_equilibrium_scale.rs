// SPDX-License-Identifier: MIT OR Apache-2.0
//! Sparse feedback systems with hundreds of variables, compared with independent dense
//! references written in this file (they share no code with the sparse solver), with explicit
//! resource accounting, interventions, typed budget refusals and uniqueness/conditioning flags.

use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionPool;
use antecedent_model::cyclic_actions::EquilibriumAction;
use antecedent_model::mechanism_expr::MechanismExpression as E;
use antecedent_model::nonlinear_equilibrium::{
    Activation, Clamp, Equation, EquilibriumError, EquilibriumModel, EquilibriumOutcome, MAX_ORDER,
    RegimeRow, SolvePolicy,
};
use antecedent_model::sparse_equilibrium::SolverBudgetCode;

fn equation(i: usize, intercept: f64, weights: Vec<f64>, activation: Activation) -> Equation {
    Equation {
        id: format!("v{i}"),
        intercept,
        weights,
        activation,
        expression: None,
        parameters: vec![],
    }
}
fn identity_covariance(n: usize) -> Vec<f64> {
    let mut cov = vec![0.0; n * n];
    for i in 0..n {
        cov[i * n + i] = 1.0;
    }
    cov
}
/// Variable `i` reads its predecessor, successor and itself on a ring.
fn ring(
    n: usize,
    predecessor: f64,
    successor: f64,
    own: f64,
    activation: Activation,
) -> EquilibriumModel {
    let equations = (0..n)
        .map(|i| {
            let mut weights = vec![0.0; n];
            weights[(i + n - 1) % n] += predecessor;
            weights[(i + 1) % n] += successor;
            weights[i] += own;
            equation(i, 0.05 * ((i % 7) as f64 - 3.0), weights, activation)
        })
        .collect();
    EquilibriumModel { equations, noise_covariance: identity_covariance(n) }
}
fn noise(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 * 0.37).sin()).collect()
}
fn weights_of(model: &EquilibriumModel) -> Vec<Vec<f64>> {
    model.equations.iter().map(|e| e.weights.clone()).collect()
}
fn intercepts_of(model: &EquilibriumModel) -> Vec<f64> {
    model.equations.iter().map(|e| e.intercept).collect()
}

// Independent dense references --------------------------------------------------------------

/// Gauss-Jordan with partial pivoting on `[a | b]`.
fn dense_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .zip(b)
        .map(|(row, rhs)| {
            let mut r = row.clone();
            r.push(*rhs);
            r
        })
        .collect();
    for j in 0..n {
        let p = (j..n).max_by(|&x, &y| m[x][j].abs().total_cmp(&m[y][j].abs())).unwrap();
        m.swap(j, p);
        let pivot = m[j][j];
        for k in j..=n {
            m[j][k] /= pivot;
        }
        for i in 0..n {
            if i != j {
                let f = m[i][j];
                if f != 0.0 {
                    for k in j..=n {
                        let t = m[j][k];
                        m[i][k] -= f * t;
                    }
                }
            }
        }
    }
    (0..n).map(|i| m[i][n]).collect()
}
fn identity_minus(w: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = w.len();
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 - w[i][j] } else { -w[i][j] }).collect())
        .collect()
}
/// Exact `||A||_inf ||A^-1||_inf` from one dense Gauss-Jordan inverse of `[A | I]`.
fn dense_condition(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let norm = a.iter().map(|r| r.iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max);
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            r
        })
        .collect();
    for j in 0..n {
        let p = (j..n).max_by(|&x, &y| m[x][j].abs().total_cmp(&m[y][j].abs())).unwrap();
        m.swap(j, p);
        let pivot = m[j][j];
        for k in 0..2 * n {
            m[j][k] /= pivot;
        }
        for i in 0..n {
            if i != j {
                let f = m[i][j];
                if f != 0.0 {
                    for k in 0..2 * n {
                        let t = m[j][k];
                        m[i][k] -= f * t;
                    }
                }
            }
        }
    }
    let inverse_norm =
        m.iter().map(|r| r[n..].iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max);
    norm * inverse_norm
}
fn dense_log_abs_det(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let mut m = a.to_vec();
    let mut total = 0.0;
    for j in 0..n {
        let p = (j..n).max_by(|&x, &y| m[x][j].abs().total_cmp(&m[y][j].abs())).unwrap();
        m.swap(j, p);
        total += m[j][j].abs().ln();
        for i in j + 1..n {
            let f = m[i][j] / m[j][j];
            for k in j..n {
                let t = m[j][k];
                m[i][k] -= f * t;
            }
        }
    }
    total
}
fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(7)
}

#[test]
fn ring_with_hundreds_of_variables_matches_the_dense_solution_and_reports_accounting() {
    let n = 320;
    let model = ring(n, 0.35, 0.25, 0.1, Activation::Identity);
    let u = noise(n);
    let solved = model.solve(&u, &[], &vec![0.0; n], SolvePolicy::default(), &ctx()).unwrap();
    let rhs: Vec<f64> = intercepts_of(&model).iter().zip(&u).map(|(c, e)| c + e).collect();
    let a = identity_minus(&weights_of(&model));
    assert!(max_diff(&solved.values, &dense_solve(&a, &rhs)) < 1e-8);
    assert!(solved.globally_unique && solved.existence_certified && solved.numerical_converged);

    let ev = &solved.evidence;
    assert_eq!((ev.free_variables, ev.block_count, ev.largest_block), (n, 1, n));
    assert_eq!(ev.cyclic_block_count, 1);
    assert_eq!(ev.jacobian_nnz, 3 * n);
    assert!(ev.contraction_iterations > 10 && ev.newton_steps == 0);
    assert_eq!(ev.factorizations, 2);
    assert!(ev.peak_factor_fill >= 2 * n && ev.peak_factor_fill < n * n / 8);
    assert!(ev.budget_checks as u64 >= ev.contraction_iterations as u64);
    assert!(ev.admitted_bytes > 0 && ev.admitted_bytes == ev.estimated_bytes);
    assert!(ev.admitted_bytes < 1 << 30);
    // The row sum 0.35 + 0.25 + 0.1 = 0.7 is the Gershgorin bound; the spectral bound cannot
    // exceed it and the contraction bound is the smaller of the two.
    assert!((ev.row_sum_bound - 0.7).abs() < 1e-9);
    assert!(ev.spectral_bound <= ev.row_sum_bound + 1e-12);
    assert!(ev.contraction_bound < 0.71);
    assert_eq!(ev.jacobian_nonsingular_basis, "contraction");
    assert_eq!(ev.unique_root_basis, "contraction");
    assert!(!ev.ill_conditioned);
    // Diagonal dominance margin of I - W: 0.9 - 0.6 = 0.3.
    assert!((ev.jacobian_diagonal_dominance_margin.unwrap() - 0.3).abs() < 1e-9);

    // Sparse conditioning estimate against the exact dense condition number.
    let exact = dense_condition(&a);
    let estimate = solved.jacobian_condition_estimate.unwrap();
    assert!(estimate <= exact * (1.0 + 1e-9) && estimate >= exact / 4.0, "{estimate} {exact}");
}

#[test]
fn nonlinear_tanh_ring_matches_an_independent_fixed_point_reference() {
    let n = 300;
    let model = ring(n, 0.3, 0.3, 0.0, Activation::Tanh);
    let u = noise(n);
    let solved = model.solve(&u, &[], &vec![5.0; n], SolvePolicy::default(), &ctx()).unwrap();
    let c = intercepts_of(&model);
    let mut x = vec![0.0; n];
    for _ in 0..400 {
        let next: Vec<f64> = (0..n)
            .map(|i| c[i] + (0.3_f64 * x[(i + n - 1) % n] + 0.3 * x[(i + 1) % n]).tanh() + u[i])
            .collect();
        x = next;
    }
    assert!(max_diff(&solved.values, &x) < 1e-8);
    assert!(solved.residual <= 1e-10);
}

#[test]
fn noncontractive_ring_is_refused_by_default_and_solved_by_sparse_newton_with_an_affine_certificate()
 {
    let n = 300;
    let model = ring(n, 1.5, 0.0, 0.0, Activation::Identity);
    let u = noise(n);
    assert!(matches!(
        model.solve(&u, &[], &vec![0.0; n], SolvePolicy::default(), &ctx()),
        Err(EquilibriumError::Uncertified { .. })
    ));
    let policy = SolvePolicy { require_contraction: false, ..SolvePolicy::default() };
    let solved = model.solve(&u, &[], &vec![0.0; n], policy, &ctx()).unwrap();
    let rhs: Vec<f64> = intercepts_of(&model).iter().zip(&u).map(|(c, e)| c + e).collect();
    let a = identity_minus(&weights_of(&model));
    assert!(max_diff(&solved.values, &dense_solve(&a, &rhs)) < 1e-8);
    assert!(solved.globally_unique);
    let ev = &solved.evidence;
    assert!(ev.newton_steps >= 1 && ev.contraction_iterations == 0);
    assert_eq!(ev.unique_root_basis, "affine_bijection");
    assert_eq!(ev.jacobian_nonsingular_basis, "factorization");
    assert!(ev.row_sum_bound >= 1.5 - 1e-9 && ev.contraction_bound >= 1.0);
    assert!(ev.factorizations >= 3);
    let exact = dense_condition(&a);
    let estimate = solved.jacobian_condition_estimate.unwrap();
    assert!(estimate <= exact * (1.0 + 1e-9) && estimate >= exact / 4.0, "{estimate} {exact}");
}

#[test]
fn spectral_bound_certifies_a_contraction_the_row_sum_cannot() {
    // Pairs with weights 1.2 and 0.3: row sums 1.2 but spectral radius sqrt(0.36) = 0.6.
    let pairs = 150;
    let n = 2 * pairs;
    let equations = (0..n)
        .map(|i| {
            let mut weights = vec![0.0; n];
            weights[i ^ 1] = if i % 2 == 0 { 1.2 } else { 0.3 };
            equation(i, 0.0, weights, Activation::Identity)
        })
        .collect();
    let model = EquilibriumModel { equations, noise_covariance: identity_covariance(n) };
    let u = noise(n);
    let solved = model.solve(&u, &[], &vec![0.0; n], SolvePolicy::default(), &ctx()).unwrap();
    let ev = &solved.evidence;
    assert!(ev.row_sum_bound > 1.19 && ev.spectral_bound < 0.61 && ev.contraction_bound < 0.61);
    assert!(ev.spectral_iterations > 0);
    assert_eq!(ev.unique_root_basis, "contraction");
    let a = identity_minus(&weights_of(&model));
    assert!(max_diff(&solved.values, &dense_solve(&a, &u)) < 1e-8);
}

#[test]
fn block_sparse_chain_of_small_cycles_is_decomposed_and_matches_dense() {
    let groups = 110;
    let n = 3 * groups;
    let equations = (0..n)
        .map(|i| {
            let g = i / 3;
            let mut weights = vec![0.0; n];
            weights[3 * g + (i + 2) % 3] = 0.3; // in-group cycle of three
            if g > 0 && i % 3 == 0 {
                weights[3 * (g - 1) + 1] = 0.2; // feed-forward coupling to the previous group
            }
            equation(i, 0.1, weights, Activation::Identity)
        })
        .collect();
    let model = EquilibriumModel { equations, noise_covariance: identity_covariance(n) };
    let u = noise(n);
    let solved = model.solve(&u, &[], &vec![0.0; n], SolvePolicy::default(), &ctx()).unwrap();
    let rhs: Vec<f64> = u.iter().map(|e| e + 0.1).collect();
    let a = identity_minus(&weights_of(&model));
    assert!(max_diff(&solved.values, &dense_solve(&a, &rhs)) < 1e-8);
    let ev = &solved.evidence;
    assert_eq!((ev.block_count, ev.largest_block, ev.cyclic_block_count), (groups, 3, groups));
    // Source-before-sink order: every group appears after the group it reads.
    let position = |v: usize| solved.blocks.iter().position(|b| b.contains(&v)).unwrap();
    for g in 1..groups {
        assert!(position(3 * (g - 1) + 1) < position(3 * g));
    }
}

#[test]
fn interventions_on_a_large_system_match_the_clamped_dense_system() {
    let n = 300;
    let model = ring(n, 0.5, 0.0, 0.0, Activation::Identity);
    let u = noise(n);
    let clamps = [
        Clamp { variable: 10, value: 2.0 },
        Clamp { variable: 150, value: -1.0 },
        Clamp { variable: 299, value: 0.5 },
    ];
    let solved = model.solve(&u, &clamps, &vec![0.0; n], SolvePolicy::default(), &ctx()).unwrap();
    let mut a = identity_minus(&weights_of(&model));
    let mut rhs: Vec<f64> = intercepts_of(&model).iter().zip(&u).map(|(c, e)| c + e).collect();
    for c in &clamps {
        a[c.variable] = (0..n).map(|j| if j == c.variable { 1.0 } else { 0.0 }).collect();
        rhs[c.variable] = c.value;
    }
    assert!(max_diff(&solved.values, &dense_solve(&a, &rhs)) < 1e-8);
    // The clamps cut the ring: no feedback remains and every variable is a singleton block.
    assert_eq!(solved.evidence.free_variables, n - 3);
    assert_eq!(solved.evidence.cyclic_block_count, 0);
    assert_eq!(solved.evidence.block_count, n - 3);
    assert_eq!(solved.evidence.jacobian_nnz, 2 * (n - 3) - 3);

    // Named actions reuse the factual disturbances: clamp, change and shock together.
    let factual = model.solve(&u, &[], &vec![0.0; n], SolvePolicy::default(), &ctx()).unwrap();
    let abducted = model.abduce(&factual.values).unwrap();
    assert!(max_diff(&abducted, &u) < 1e-8);
    let plan = model
        .compile_actions(&[
            EquilibriumAction::Clamp { mechanism: "v10".into(), value: 2.0 },
            EquilibriumAction::AdditiveShock { mechanism: "v77".into(), delta: 0.75 },
            EquilibriumAction::ChangeMechanism {
                mechanism: "v200".into(),
                intercept: Some(1.0),
                weights: None,
                activation: None,
            },
        ])
        .unwrap();
    let EquilibriumOutcome::Solved(acted) =
        plan.solve(&abducted, &vec![0.0; n], SolvePolicy::default(), &ctx()).unwrap()
    else {
        panic!("a contractive ring with one clamp must solve");
    };
    let mut a = identity_minus(&weights_of(&model));
    let mut rhs: Vec<f64> = intercepts_of(&model).iter().zip(&u).map(|(c, e)| c + e).collect();
    a[10] = (0..n).map(|j| if j == 10 { 1.0 } else { 0.0 }).collect();
    rhs[10] = 2.0;
    rhs[77] += 0.75;
    rhs[200] += 1.0 - model.equations[200].intercept;
    assert!(max_diff(&acted.values, &dense_solve(&a, &rhs)) < 1e-8);
    assert_eq!(acted.evidence.free_variables, n - 1);
}

#[test]
fn compositional_expressions_work_beyond_the_old_variable_cap() {
    let n = 200;
    let expression = |i: usize| E::Multiply {
        factors: vec![
            E::Parameter { parameter: 0 },
            E::Tanh {
                argument: Box::new(E::Add {
                    terms: vec![
                        E::Multiply {
                            factors: vec![
                                E::Parameter { parameter: 1 },
                                E::Variable { variable: (i + n - 1) % n },
                            ],
                        },
                        E::Multiply {
                            factors: vec![
                                E::Parameter { parameter: 2 },
                                E::Variable { variable: (i + 1) % n },
                            ],
                        },
                    ],
                }),
            },
        ],
    };
    assert!(expression(0).compile(n, 3).is_ok());
    let equations = (0..n)
        .map(|i| Equation {
            id: format!("v{i}"),
            intercept: 0.1,
            weights: vec![0.0; n],
            activation: Activation::Identity,
            expression: Some(expression(i)),
            parameters: vec![0.5, 0.6, 0.4],
        })
        .collect();
    let model = EquilibriumModel { equations, noise_covariance: identity_covariance(n) };
    let u = noise(n);
    let solved = model.solve(&u, &[], &vec![0.0; n], SolvePolicy::default(), &ctx()).unwrap();
    let mut x = vec![0.0; n];
    for _ in 0..300 {
        let next: Vec<f64> = (0..n)
            .map(|i| 0.1 + 0.5 * (0.6_f64 * x[(i + n - 1) % n] + 0.4 * x[(i + 1) % n]).tanh() + u[i])
            .collect();
        x = next;
    }
    assert!(max_diff(&solved.values, &x) < 1e-8);
    assert_eq!(solved.evidence.largest_block, n);
    assert!(solved.evidence.row_sum_bound >= 0.5 - 1e-9 && solved.evidence.row_sum_bound < 1.0);
    // Beyond the model order the expression is refused with the order code, not a silent cap.
    let refusal = expression(0).compile(MAX_ORDER + 1, 3).unwrap_err();
    assert_eq!(refusal.budget_code(), Some(SolverBudgetCode::ModelOrder));
}

#[test]
fn sparse_likelihood_log_determinant_matches_the_dense_reference() {
    let n = 150;
    let model = ring(n, 0.35, 0.25, 0.1, Activation::Identity);
    let values: Vec<f64> = (0..n).map(|i| (i as f64 * 0.11).cos()).collect();
    let got = model.log_likelihood(&[RegimeRow { values: values.clone(), clamps: vec![] }], &ctx());
    let a = identity_minus(&weights_of(&model));
    let mut z = 0.0;
    for i in 0..n {
        let mean: f64 = model.equations[i].intercept
            + (0..n).map(|j| model.equations[i].weights[j] * values[j]).sum::<f64>();
        z += (values[i] - mean).powi(2);
    }
    let expected = dense_log_abs_det(&a) - 0.5 * (z + n as f64 * (2.0 * std::f64::consts::PI).ln());
    assert!((got.unwrap() - expected).abs() < 1e-8);
}

// Resource admission --------------------------------------------------------------------------

fn budget_code(model: &EquilibriumModel, policy: SolvePolicy) -> Option<SolverBudgetCode> {
    let n = model.order();
    model.solve(&noise(n), &[], &vec![0.0; n], policy, &ctx()).unwrap_err().budget_code()
}

#[test]
fn every_exceeded_budget_has_its_own_typed_refusal() {
    let n = 200;
    let contractive = ring(n, 0.35, 0.25, 0.1, Activation::Identity);
    let noncontractive = ring(n, 1.5, 0.0, 0.0, Activation::Identity);
    let newton = SolvePolicy { require_contraction: false, ..SolvePolicy::default() };
    let base = SolvePolicy::default();
    assert_eq!(
        budget_code(&contractive, SolvePolicy { max_nnz: 10, ..base }),
        Some(SolverBudgetCode::JacobianNnz)
    );
    assert_eq!(
        budget_code(&contractive, SolvePolicy { max_block_size: 50, ..base }),
        Some(SolverBudgetCode::BlockSize)
    );
    assert_eq!(
        budget_code(&contractive, SolvePolicy { max_bytes: 1000, ..base }),
        Some(SolverBudgetCode::Memory)
    );
    assert_eq!(
        budget_code(&contractive, SolvePolicy { max_total_iterations: 1, ..base }),
        Some(SolverBudgetCode::IterationBudget)
    );
    assert_eq!(
        budget_code(&contractive, SolvePolicy { max_wall_clock_ms: Some(0), ..base }),
        Some(SolverBudgetCode::WallClock)
    );
    assert_eq!(
        budget_code(&noncontractive, SolvePolicy { max_factor_fill: 300, ..newton }),
        Some(SolverBudgetCode::FactorFill)
    );
    // The conditioning factorization is also bounded: a contractive solve whose factor does
    // not fit still returns its root, with the conditioning explicitly marked not attempted.
    let tight = SolvePolicy { max_factor_fill: 450, ..base };
    let solved = contractive.solve(&noise(n), &[], &vec![0.0; n], tight, &ctx());
    let solved = solved.unwrap();
    assert!(solved.jacobian_condition_estimate.is_none());
    assert_eq!(solved.evidence.jacobian_nonsingular_basis, "factor_fill_budget");
    // Distinct codes, distinct identifiers.
    let ids = [
        SolverBudgetCode::ModelOrder,
        SolverBudgetCode::JacobianNnz,
        SolverBudgetCode::BlockSize,
        SolverBudgetCode::Memory,
        SolverBudgetCode::FactorFill,
        SolverBudgetCode::IterationBudget,
        SolverBudgetCode::WallClock,
    ]
    .map(SolverBudgetCode::as_str);
    let unique: std::collections::BTreeSet<_> = ids.iter().collect();
    assert_eq!(unique.len(), ids.len());
}

#[test]
fn cancellation_memory_admission_and_malformed_policy_are_not_budget_codes() {
    let n = 200;
    let model = ring(n, 0.35, 0.25, 0.1, Activation::Identity);
    let u = noise(n);
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    let error =
        model.solve(&u, &[], &vec![0.0; n], SolvePolicy::default(), &cancelled).unwrap_err();
    assert!(matches!(error, EquilibriumError::Resource(_)));
    assert_eq!(error.budget_code(), None);

    let small = ExecutionContext::for_tests(1).with_admission(AdmissionPool::fixed(1, 1000));
    let error = model.solve(&u, &[], &vec![0.0; n], SolvePolicy::default(), &small).unwrap_err();
    assert!(matches!(error, EquilibriumError::Resource(_)));
    assert_eq!(error.budget_code(), None);

    let bad = SolvePolicy { max_nnz: 0, ..SolvePolicy::default() };
    assert!(matches!(
        model.solve(&u, &[], &vec![0.0; n], bad, &ctx()),
        Err(EquilibriumError::Invalid(_))
    ));
}

#[test]
fn numerical_failures_stay_retained_outcomes_while_budget_refusals_abort() {
    let n = 200;
    let model = ring(n, 0.35, 0.25, 0.1, Activation::Identity);
    let u = noise(n);
    let one = SolvePolicy { max_iterations: 1, ..SolvePolicy::default() };
    let outcome = model.solve_outcome(&u, &[], &vec![0.0; n], one, &ctx()).unwrap();
    assert!(matches!(outcome, EquilibriumOutcome::Failed(f) if f.failure == "nonconvergence"));
    let capped = SolvePolicy { max_total_iterations: 1, ..SolvePolicy::default() };
    let aborted = model.solve_outcome(&u, &[], &vec![0.0; n], capped, &ctx());
    assert_eq!(aborted.unwrap_err().budget_code(), Some(SolverBudgetCode::IterationBudget));
}

// Uniqueness and conditioning flags --------------------------------------------------------------

#[test]
fn multiple_equilibria_are_flagged_as_not_unique() {
    // 100 independent bistable pairs: x_a = tanh(2 x_b), x_b = tanh(2 x_a).
    let pairs = 100;
    let n = 2 * pairs;
    let equations = (0..n)
        .map(|i| {
            let mut weights = vec![0.0; n];
            weights[i ^ 1] = 2.0;
            equation(i, 0.0, weights, Activation::Tanh)
        })
        .collect();
    let model = EquilibriumModel { equations, noise_covariance: identity_covariance(n) };
    let policy = SolvePolicy { require_contraction: false, ..SolvePolicy::default() };
    let zero = vec![0.0; n];
    let positive = model.solve(&zero, &[], &vec![1.0; n], policy, &ctx()).unwrap();
    let negative = model.solve(&zero, &[], &vec![-1.0; n], policy, &ctx()).unwrap();
    assert!(positive.values.iter().all(|v| *v > 0.9) && negative.values.iter().all(|v| *v < -0.9));
    for result in [&positive, &negative] {
        assert!(!result.globally_unique && !result.existence_certified);
        assert!(result.numerical_converged);
        assert_eq!(result.evidence.unique_root_basis, "none");
        assert_eq!(result.evidence.cyclic_block_count, pairs);
        assert!(result.evidence.contraction_bound >= 1.0);
        assert!(result.evidence.newton_steps > 0);
    }
    assert!(matches!(
        model.log_likelihood(&[RegimeRow { values: positive.values, clamps: vec![] }], &ctx()),
        Err(EquilibriumError::Uncertified { .. })
    ));
}

#[test]
fn nearly_singular_systems_are_flagged_ill_conditioned_and_singular_ones_are_not_certified() {
    let n = 300;
    let policy = SolvePolicy { require_contraction: false, ..SolvePolicy::default() };
    let zero = vec![0.0; n];
    // I - w P on a ring has determinant 1 - w^n; w = 1 + 1e-13 is nonsingular but extremely
    // ill conditioned. Zero intercepts and disturbance keep the root at the origin.
    let mut near = ring(n, 1.0 + 1e-13, 0.0, 0.0, Activation::Identity);
    for e in &mut near.equations {
        e.intercept = 0.0;
    }
    let result = near.solve(&zero, &[], &zero, policy, &ctx()).unwrap();
    let estimate = result.jacobian_condition_estimate.unwrap();
    assert!(estimate > 1e9, "{estimate}");
    assert!(result.evidence.ill_conditioned);
    assert_eq!(result.evidence.jacobian_nonsingular_basis, "factorization");

    // Exactly w = 1: I - P is singular, so no certificate and no conditioning is claimed.
    let mut singular = ring(n, 1.0, 0.0, 0.0, Activation::Identity);
    for e in &mut singular.equations {
        e.intercept = 0.0;
    }
    let result = singular.solve(&zero, &[], &zero, policy, &ctx()).unwrap();
    assert!(result.jacobian_condition_estimate.is_none());
    assert!(!result.globally_unique && !result.evidence.ill_conditioned);
    assert_eq!(result.evidence.jacobian_nonsingular_basis, "singular");
    assert_eq!(result.evidence.unique_root_basis, "none");
    // A well-conditioned contractive system is not flagged.
    let fine = ring(n, 0.3, 0.3, 0.0, Activation::Identity);
    let result = fine.solve(&noise(n), &[], &zero, SolvePolicy::default(), &ctx()).unwrap();
    assert!(!result.evidence.ill_conditioned);
    // A lower warning threshold flags the same contractive system, which is a policy choice.
    let strict = SolvePolicy { condition_warning: 1.5, ..SolvePolicy::default() };
    let result = fine.solve(&noise(n), &[], &zero, strict, &ctx()).unwrap();
    assert!(result.evidence.ill_conditioned);
}

// Retained NumPy/LAPACK reference -----------------------------------------------------------------

#[test]
fn solutions_match_the_retained_numpy_reference_document() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/gcm/sparse_feedback_equilibrium/expected.json"
    );
    let document: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let policy = SolvePolicy { require_contraction: false, ..SolvePolicy::default() };
    for case in document["cases"].as_array().unwrap() {
        let number = |key: &str| case[key].as_f64().unwrap();
        let n = case["order"].as_u64().unwrap() as usize;
        let mut model = ring(
            n,
            number("predecessor"),
            number("successor"),
            number("own"),
            Activation::Identity,
        );
        let u = noise(n);
        let clamps: Vec<Clamp> = case["clamps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| Clamp {
                variable: c[0].as_u64().unwrap() as usize,
                value: c[1].as_f64().unwrap(),
            })
            .collect();
        // The reference draws noise sin(0.37 i) and intercepts 0.05 * ((i % 7) - 3): `ring`
        // and `noise` use exactly those rules, so the same linear system is solved here.
        for (i, e) in model.equations.iter_mut().enumerate() {
            e.intercept = 0.05 * ((i % 7) as f64 - 3.0);
        }
        let solved = model.solve(&u, &clamps, &vec![0.0; n], policy, &ctx()).unwrap();
        let head: Vec<f64> =
            case["values_head"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        assert!(max_diff(&solved.values[..head.len()], &head) < 1e-8, "{}", case["case"]);
        let sum: f64 = solved.values.iter().sum();
        assert!((sum - number("values_sum")).abs() < 1e-7 * (1.0 + sum.abs()), "{}", case["case"]);
        let max_abs = solved.values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!((max_abs - number("values_max_abs")).abs() < 1e-8, "{}", case["case"]);
        let exact = number("infinity_condition");
        let estimate = solved.jacobian_condition_estimate.unwrap();
        assert!(estimate <= exact * (1.0 + 1e-8) && estimate >= exact / 4.0, "{}", case["case"]);
        if clamps.is_empty() {
            let ev = &solved.evidence;
            assert!((ev.row_sum_bound - number("abs_weight_row_sum")).abs() < 1e-8);
            assert!(ev.spectral_bound >= number("abs_weight_spectral_radius") * (1.0 - 1e-9));
        }
    }
}

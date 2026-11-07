use super::{
    CompileBudget, GaussianDisturbance, LinearRouteSet, LinearSupportRoute, SccCompileError,
    block_residual, compile_gaussian_equilibrium, compile_routed_gaussian_equilibrium,
    compile_routed_with, gaussian_equilibrium_admission, gaussian_equilibrium_route_work,
    plan_bytes, route_linear_support,
};
use crate::cyclic_component::{CyclicComponent, DAG_EQUILIBRIUM_ATTEMPT, SCC_COMPILE_ATTEMPT};
use crate::linear_solution::{
    FREE_BLOCK_EQUATION, HardClamp, IdentificationStatus, LinearSolutionError, LinearStructuralMap,
    NONZERO_DIAGONAL_NEXT_ATTEMPT, PhysicalStability, SolverConvergence, Uniqueness, assess,
};

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-8
}

fn map(coefficients: Vec<f64>, intercept: Vec<f64>) -> LinearStructuralMap {
    LinearStructuralMap::try_new(coefficients, intercept).expect("fixture map")
}

/// Component admitted for `model` under the family certificate it yields. Only models
/// with `ρ(|K|) < 1` can be wrapped.
fn component_for(model: &LinearStructuralMap) -> CyclicComponent {
    let family = assess(model, &HardClamp::none(model.dimension()).expect("no clamp"))
        .expect("assessment")
        .family;
    CyclicComponent::try_new("feedback", "source-measurements", &["solve-note"], model, &family)
        .expect("component")
}

fn compile(
    model: &LinearStructuralMap,
    mean: Vec<f64>,
    covariance: Vec<f64>,
    clamp: &HardClamp,
    values: &[f64],
) -> super::CompiledEquilibrium {
    let disturbance = GaussianDisturbance::try_new(mean, covariance).expect("disturbance");
    compile_gaussian_equilibrium(
        &component_for(model),
        model,
        &disturbance,
        clamp,
        values,
        CompileBudget::unlimited(),
    )
    .expect("equilibrium")
}

#[test]
fn two_node_cycle_matches_the_hand_solution() {
    // A = 0.4 B + ε_A, B = 0.5 A + ε_B. Indices: A = 0, B = 1.
    // I - K = [[1, -0.4], [-0.5, 1]], det = 0.8.
    // (I - K)^{-1} = [[1.25, 0.5], [0.625, 1.25]].
    // With μ_ε = (1, 0) and c = 0, μ = (1.25, 0.625).
    // With Σ = I, Cov = (I - K)^{-1}(I - K)^{-T}
    //         = [[1.8125, 1.40625], [1.40625, 1.953125]].
    // ||I - K||∞ = 1.5, ||(I - K)^{-1}||∞ = 1.875, κ∞ = 2.8125.
    let model = map(vec![0.0, 0.4, 0.5, 0.0], vec![0.0, 0.0]);
    let clamp = HardClamp::none(2).expect("no clamp");
    let cost = gaussian_equilibrium_admission(&model, &clamp).expect("admission");
    // n = 2, one block of order 2, no clamps:
    // routing 2n² = 8, PSD n³ = 8, residual n² = 4,
    // block = m³ + m²(m + n) + m = 8 + 16 + 2 = 26, condition = m² + 16 = 20.
    assert_eq!(cost.work(), 66);
    // 8 · (15 n² + 16 n) = 8 · 92.
    assert_eq!(cost.bytes(), 736);

    let law = compile(&model, vec![1.0, 0.0], vec![1.0, 0.0, 0.0, 1.0], &clamp, &[]);
    assert_eq!(law.component_id(), "feedback");
    assert_eq!(law.free_block_equation(), FREE_BLOCK_EQUATION);
    assert_eq!(law.compile_attempt(), SCC_COMPILE_ATTEMPT);
    assert_eq!(component_for(&model).scc_compile_attempt(), SCC_COMPILE_ATTEMPT);
    assert_eq!(law.identification(), IdentificationStatus::NotMarkedUnidentified);
    assert_eq!(law.physical_stability(), PhysicalStability::Unclaimed);
    assert_eq!(law.solver(), SolverConvergence::NotRun);
    assert_eq!(law.admission(), cost);
    let wrapped = component_for(&model);
    assert!(!wrapped.evidence_may_certify(wrapped.source_id()));

    assert!(close(law.mean()[0], 1.25));
    assert!(close(law.mean()[1], 0.625));
    assert!(close(law.covariance()[0], 1.8125));
    assert!(close(law.covariance()[1], 1.40625));
    assert!(close(law.covariance()[3], 1.953_125));
    assert!(close(law.covariance()[1] / law.covariance()[0], 45.0 / 58.0));
    assert!(law.residual_inf() < 1e-8);
    assert_eq!(law.blocks().len(), 1);
    assert_eq!(law.blocks()[0].members, vec![0, 1]);
    assert!(law.blocks()[0].exact);
    assert!(close(law.blocks()[0].kappa_inf, 2.8125));
    assert!(law.blocks()[0].residual_inf < 1e-8);
}

#[test]
fn cross_block_disturbance_covariance_is_kept_without_a_structural_edge() {
    // Two singleton components and Σ_01 = 0.3. Dropping the cross entry would
    // report Cov(X_0, X_1) = 0.
    let model = map(vec![0.0, 0.0, 0.0, 0.0], vec![0.0, 0.0]);
    let clamp = HardClamp::none(2).expect("no clamp");
    let law = compile(&model, vec![0.0, 0.0], vec![1.0, 0.3, 0.3, 1.0], &clamp, &[]);
    assert_eq!(law.blocks().len(), 2);
    assert_eq!(law.blocks()[0].members, vec![0]);
    assert_eq!(law.blocks()[1].members, vec![1]);
    assert!(close(law.covariance()[0], 1.0));
    assert!(close(law.covariance()[1], 0.3));
    assert!(close(law.covariance()[3], 1.0));
    assert!(law.residual_inf() < 1e-8);
}

#[test]
fn chain_keeps_structural_and_disturbance_cross_covariance() {
    // X0 = 0.5 + ε0, X1 = 0.25 + 0.5 X0 + ε1.
    // μ_ε = (1, -2), Σ = [[1, 0.3], [0.3, 1]].
    // μ = (1.5, -1), Cov(X0, X1) = 0.8, Var(X1) = 1.55.
    // Independence of the disturbances would give Cov = 0.5 and Var(X1) = 1.25.
    let model = map(vec![0.0, 0.0, 0.5, 0.0], vec![0.5, 0.25]);
    let clamp = HardClamp::none(2).expect("no clamp");
    let law = compile(&model, vec![1.0, -2.0], vec![1.0, 0.3, 0.3, 1.0], &clamp, &[]);
    assert_eq!(law.blocks()[0].members, vec![0]);
    assert_eq!(law.blocks()[1].members, vec![1]);
    assert!(close(law.mean()[0], 1.5));
    assert!(close(law.mean()[1], -1.0));
    assert!(close(law.covariance()[0], 1.0));
    assert!(close(law.covariance()[1], 0.8));
    assert!(close(law.covariance()[3], 1.55));
    assert!(law.residual_inf() < 1e-8);
}

#[test]
fn upstream_parent_of_a_two_node_cycle_matches_the_hand_solution() {
    // X0 = ε0, X1 = 0.2 X0 + 0.4 X2 + ε1, X2 = 0.5 X1 + ε2, Σ = I, c = 0.
    // The cycle inverse is [[1.25, 0.5], [0.625, 1.25]], so
    // X1 = 0.25 ε0 + 1.25 ε1 + 0.5 ε2,
    // X2 = 0.125 ε0 + 0.625 ε1 + 1.25 ε2.
    // Cov(X0, X1) = 0.25, Cov(X0, X2) = 0.125,
    // Var(X1) = 1.875, Cov(X1, X2) = 1.4375, Var(X2) = 1.96875.
    let model = map(vec![0.0, 0.0, 0.0, 0.2, 0.0, 0.4, 0.0, 0.5, 0.0], vec![0.0, 0.0, 0.0]);
    let clamp = HardClamp::none(3).expect("no clamp");
    let law = compile(
        &model,
        vec![0.0; 3],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &clamp,
        &[],
    );
    assert_eq!(law.blocks().len(), 2);
    assert_eq!(law.blocks()[0].members, vec![0]);
    assert_eq!(law.blocks()[1].members, vec![1, 2]);
    let cov = law.covariance();
    assert!(close(cov[0], 1.0));
    assert!(close(cov[1], 0.25));
    assert!(close(cov[2], 0.125));
    assert!(close(cov[4], 1.875));
    assert!(close(cov[5], 1.4375));
    assert!(close(cov[8], 1.96875));
    assert!(law.residual_inf() < 1e-8);
    assert!(law.blocks()[1].exact);
    assert!(close(law.blocks()[1].kappa_inf, 2.8125));
}

#[test]
fn block_residual_refuses_overflow_from_finite_operands() {
    assert!(matches!(
        block_residual(&[1.0], &[f64::MAX], 1, &[-f64::MAX], 1),
        Err(SccCompileError::NonFiniteEquilibrium)
    ));
}

#[test]
fn hard_clamp_keeps_the_marginal_free_disturbance() {
    // do(A = 2) on A = 0.4 B + ε_A, B = 0.5 A + ε_B, with Cov(ε_A, ε_B) = 0.3.
    // B = 1 + ε_B. Var(B) stays 1, not the conditional variance 0.91.
    let model = map(vec![0.0, 0.4, 0.5, 0.0], vec![0.0, 0.0]);
    let clamp = HardClamp::try_new(2, &[0]).expect("clamp A");
    let law = compile(&model, vec![0.0, 0.0], vec![1.0, 0.3, 0.3, 1.0], &clamp, &[2.0]);
    assert!(close(law.mean()[0], 2.0));
    assert!(close(law.mean()[1], 1.0));
    assert!(close(law.covariance()[0], 0.0));
    assert!(close(law.covariance()[1], 0.0));
    assert!(close(law.covariance()[3], 1.0));
    assert_eq!(law.blocks().len(), 1);
    assert_eq!(law.blocks()[0].members, vec![1]);
    assert!(law.residual_inf() < 1e-8);

    let both = HardClamp::try_new(2, &[1, 0]).expect("clamp both");
    let assigned = compile(&model, vec![0.0, 0.0], identity(2), &both, &[4.0, 5.0]);
    assert!(assigned.blocks().is_empty());
    assert!(close(assigned.mean()[0], 4.0));
    assert!(close(assigned.mean()[1], 5.0));
    assert!(assigned.covariance().iter().all(|entry| close(*entry, 0.0)));
    assert!(close(assigned.residual_inf(), 0.0));
}

/// Compile without the component gate, for models the family certificate would not admit.
fn compile_unbound(
    model: &LinearStructuralMap,
    covariance: Vec<f64>,
    clamp: &HardClamp,
    values: &[f64],
) -> Result<super::CompiledEquilibrium, SccCompileError> {
    let n = model.dimension();
    let noise = GaussianDisturbance::try_new(vec![0.0; n], covariance).expect("noise");
    super::compile_checked("unbound", model, &noise, clamp, values, CompileBudget::unlimited())
}

#[test]
fn singular_blocks_are_refused() {
    let cycle = map(vec![0.0, 1.0, 1.0, 0.0], vec![0.0, 0.0]);
    let err = compile_unbound(&cycle, identity(2), &HardClamp::none(2).expect("none"), &[])
        .expect_err("singular cycle");
    assert!(
        matches!(err, SccCompileError::NumericallySingularBlock { members } if members == [0, 1])
    );

    // A = B + C + ε_A, B = A + ε_B, C = A + ε_C. do(C = 0) leaves a singular pair.
    let overlapping = map(vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]);
    let clamp_c = HardClamp::try_new(3, &[2]).expect("clamp C");
    let err = compile_unbound(&overlapping, identity(3), &clamp_c, &[0.0])
        .expect_err("singular free block");
    assert!(
        matches!(err, SccCompileError::NumericallySingularBlock { members } if members == [0, 1])
    );

    let mut coefficients = vec![0.0; 16];
    for node in 0..4 {
        coefficients[node * 4 + (node + 1) % 4] = 1.0;
    }
    let four = map(coefficients, vec![0.0; 4]);
    let err = compile_unbound(&four, identity(4), &HardClamp::none(4).expect("none"), &[])
        .expect_err("singular 4-cycle");
    assert!(matches!(
        err,
        SccCompileError::NumericallySingularBlock { members } if members == [0, 1, 2, 3]
    ));
}

#[test]
fn assess_and_compile_agree_on_a_pivot_below_the_floor() {
    // I - K = [[1, -1], [-(1 + 1e-13), 1]]: det = -1e-13 is not zero, but the second
    // pivot after row exchange is about 1e-13, below the 1e-12 floor.
    let near = map(vec![0.0, 1.0, 1.0 + 1.0e-13, 0.0], vec![0.0, 0.0]);
    let clamp = HardClamp::none(2).expect("none");
    let report = assess(&near, &clamp).expect("assessment");
    assert_eq!(report.original.uniqueness, Uniqueness::NumericallySingular { members: vec![0, 1] });
    let err = compile_unbound(&near, identity(2), &clamp, &[]).expect_err("compile");
    assert_eq!(err, SccCompileError::NumericallySingularBlock { members: vec![0, 1] });
    // Neither route may call the block clean: the family bound also refuses it.
    assert!(!report.family.is_admitted());

    // A pivot above the floor is accepted by both.
    let ok = map(vec![0.0, 1.0, 1.0 - 1.0e-9, 0.0], vec![0.0, 0.0]);
    let report = assess(&ok, &clamp).expect("assessment");
    assert!(matches!(report.original.uniqueness, Uniqueness::IllConditioned { .. }));
    let law = compile_unbound(&ok, identity(2), &clamp, &[]).expect("compile");
    assert!(law.blocks()[0].kappa_inf > 1.0e9);
}

#[test]
fn a_component_is_bound_to_its_model() {
    let model = map(vec![0.0, 0.4, 0.5, 0.0], vec![0.0, 0.0]);
    let wrapped = component_for(&model);
    let noise = GaussianDisturbance::try_new(vec![0.0; 2], identity(2)).expect("noise");
    let clamp = HardClamp::none(2).expect("none");

    let reweighted = map(vec![0.0, 0.4, 0.6, 0.0], vec![0.0, 0.0]);
    let err = compile_gaussian_equilibrium(
        &wrapped,
        &reweighted,
        &noise,
        &clamp,
        &[],
        CompileBudget::unlimited(),
    )
    .expect_err("different coefficients");
    assert_eq!(err, SccCompileError::ComponentModelMismatch { component: "feedback".to_owned() });

    let larger = map(vec![0.0; 9], vec![0.0; 3]);
    let noise3 = GaussianDisturbance::try_new(vec![0.0; 3], identity(3)).expect("noise");
    let err = compile_gaussian_equilibrium(
        &wrapped,
        &larger,
        &noise3,
        &HardClamp::none(3).expect("none"),
        &[],
        CompileBudget::unlimited(),
    )
    .expect_err("different order");
    assert_eq!(err, SccCompileError::ComponentDimension { component: 2, model: 3 });
}

#[test]
fn a_negative_variance_hidden_by_a_large_one_is_refused() {
    // Σ = diag(1e6, -1e-9). A tolerance scaled by the largest entry (1e6) would pass
    // it and emit Var = -1e-9.
    let model = map(vec![0.0, 0.0, 0.0, 0.0], vec![0.0, 0.0]);
    let noise = GaussianDisturbance::try_new(vec![0.0; 2], vec![1.0e6, 0.0, 0.0, -1.0e-9])
        .expect("finite symmetric");
    let err = compile_gaussian_equilibrium(
        &component_for(&model),
        &model,
        &noise,
        &HardClamp::none(2).expect("none"),
        &[],
        CompileBudget::unlimited(),
    )
    .expect_err("negative variance");
    assert_eq!(err, SccCompileError::NotPositiveSemidefinite { index: 1 });

    // A badly scaled but positive covariance is still accepted.
    let scaled = compile(
        &model,
        vec![0.0; 2],
        vec![1.0e6, 0.0, 0.0, 1.0e-9],
        &HardClamp::none(2).expect("none"),
        &[],
    );
    assert!((scaled.covariance()[3] - 1.0e-9).abs() < 1.0e-18);
}

#[test]
fn weight_half_four_cycle_has_a_small_residual_and_a_condition_bound() {
    let mut coefficients = vec![0.0; 16];
    for node in 0..4 {
        coefficients[node * 4 + (node + 1) % 4] = 0.5;
    }
    let model = map(coefficients, vec![0.0; 4]);
    let mut mean = vec![0.0; 4];
    mean[0] = 1.0;
    let law = compile(&model, mean, identity(4), &HardClamp::none(4).expect("no clamp"), &[]);
    assert_eq!(law.blocks().len(), 1);
    assert_eq!(law.blocks()[0].members, vec![0, 1, 2, 3]);
    assert!(!law.blocks()[0].exact);
    assert!(law.blocks()[0].kappa_inf.is_finite());
    assert!(law.blocks()[0].kappa_inf > 0.0);
    assert!(law.residual_inf() < 1e-8);
    assert_eq!(law.identification(), IdentificationStatus::NotMarkedUnidentified);
}

#[test]
fn a_third_order_block_uses_the_hager_higham_estimate() {
    let mut coefficients = vec![0.0; 9];
    coefficients[2] = 0.1;
    coefficients[3] = 0.1;
    coefficients[7] = 0.1;
    let model = map(coefficients, vec![0.0; 3]);
    let clamp = HardClamp::none(3).expect("none");
    let law = compile(&model, vec![0.0; 3], identity(3), &clamp, &[]);
    assert_eq!(law.blocks().len(), 1);
    assert_eq!(law.blocks()[0].members, vec![0, 1, 2]);
    assert!(!law.blocks()[0].exact);
    let matrix = [
        1.0, 0.0, -0.1, //
        -0.1, 1.0, 0.0, //
        0.0, -0.1, 1.0,
    ];
    assert!(close(law.blocks()[0].kappa_inf, infinity_condition(&matrix)));
}

fn infinity_condition(matrix: &[f64]) -> f64 {
    let mut factor = matrix.to_vec();
    let mut inverse = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    for pivot in 0..3 {
        let mut scale = factor[pivot * 3 + pivot];
        for column in 0..3 {
            factor[pivot * 3 + column] /= scale;
            inverse[pivot * 3 + column] /= scale;
        }
        for row in 0..3 {
            if row == pivot {
                continue;
            }
            scale = factor[row * 3 + pivot];
            for column in 0..3 {
                factor[row * 3 + column] -= scale * factor[pivot * 3 + column];
                inverse[row * 3 + column] -= scale * inverse[pivot * 3 + column];
            }
        }
    }
    let norm = |rows: &[f64]| {
        rows.chunks(3).fold(0.0_f64, |best, row| {
            best.max(row.iter().fold(0.0_f64, |acc, entry| acc + entry.abs()))
        })
    };
    norm(matrix) * norm(&inverse)
}

#[test]
fn nonzero_diagonal_names_the_next_attempt() {
    let diagonal = LinearStructuralMap::try_new(vec![0.2, 0.0, 0.0, 0.0], vec![0.0, 0.0]);
    let err = diagonal.expect_err("nonzero diagonal");
    assert_eq!(
        err,
        LinearSolutionError::NonzeroDiagonal {
            node: 0,
            next_attempt: NONZERO_DIAGONAL_NEXT_ATTEMPT,
        }
    );
    assert!(err.to_string().contains("next attempt"));
    assert!(err.to_string().contains(NONZERO_DIAGONAL_NEXT_ATTEMPT));
}

#[test]
fn non_finite_indefinite_and_budget_shortfalls_are_refused() {
    let non_finite = GaussianDisturbance::try_new(vec![f64::NAN, 0.0], identity(2));
    assert!(matches!(non_finite, Err(SccCompileError::NonFiniteDisturbanceMean { node: 0 })));
    let asymmetric = GaussianDisturbance::try_new(vec![0.0, 0.0], vec![1.0, 0.2, 0.4, 1.0]);
    assert!(matches!(asymmetric, Err(SccCompileError::AsymmetricCovariance { row: 0, column: 1 })));

    let model = map(vec![0.0, 0.4, 0.5, 0.0], vec![0.0, 0.0]);
    let clamp = HardClamp::none(2).expect("no clamp");
    let noise = GaussianDisturbance::try_new(vec![0.0, 0.0], identity(2)).expect("noise");
    let indefinite = GaussianDisturbance::try_new(vec![0.0, 0.0], vec![1.0, 2.0, 2.0, 1.0])
        .expect("indefinite shape is finite and symmetric");
    let err = compile_gaussian_equilibrium(
        &component_for(&model),
        &model,
        &indefinite,
        &clamp,
        &[],
        CompileBudget::unlimited(),
    )
    .expect_err("indefinite covariance");
    assert!(matches!(err, SccCompileError::NotPositiveSemidefinite { index: 1 }));

    let cost = gaussian_equilibrium_admission(&model, &clamp).expect("admission");
    let short_work = compile_gaussian_equilibrium(
        &component_for(&model),
        &model,
        &noise,
        &clamp,
        &[],
        CompileBudget { max_work: cost.work() - 1, max_bytes: u64::MAX },
    )
    .expect_err("work budget");
    assert_eq!(
        short_work,
        SccCompileError::Budget {
            resource: "work",
            requested: cost.work(),
            limit: cost.work() - 1,
        }
    );
    let short_bytes = compile_gaussian_equilibrium(
        &component_for(&model),
        &model,
        &noise,
        &clamp,
        &[],
        CompileBudget { max_work: u64::MAX, max_bytes: cost.bytes() - 1 },
    )
    .expect_err("byte budget");
    assert!(matches!(
        short_bytes,
        SccCompileError::Budget { resource: "bytes", requested, limit }
            if requested == cost.bytes() && limit == cost.bytes() - 1
    ));
}

#[test]
fn routing_and_compilation_are_charged_against_one_work_budget() {
    let cycle = map(vec![0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0], vec![0.0; 3]);
    let clamp = HardClamp::none(3).unwrap();
    let noise = GaussianDisturbance::try_new(vec![0.0; 3], identity(3)).unwrap();
    let total = gaussian_equilibrium_admission(&cycle, &clamp).unwrap();
    let route = 2 * 3 * 3;
    let compile = |max_work| {
        compile_routed_gaussian_equilibrium(
            &component_for(&cycle),
            &cycle,
            &noise,
            &clamp,
            &[],
            CompileBudget { max_work, max_bytes: u64::MAX },
        )
    };
    // Each part fits alone; only the sum is over the budget.
    let factorization = total.work() - route;
    assert!(route < total.work() && factorization < total.work());
    assert!(matches!(
        compile(total.work() - 1),
        Err(SccCompileError::Budget { resource: "work", requested, .. })
            if requested == total.work()
    ));
    let compiled = compile(total.work()).unwrap();
    assert_eq!(compiled.equilibrium().admission, total);
}

#[test]
fn plan_bytes_cover_the_peak_dense_storage() {
    // 14 n² transient floats and n² retained, then 16 n linear floats, times 8 bytes.
    for n in [1_u64, 2, 7, 64] {
        assert_eq!(plan_bytes(n).unwrap(), 8 * (15 * n * n + 16 * n));
    }
    assert!(plan_bytes(u64::MAX).is_err());
}

#[test]
fn a_held_route_is_neither_routed_nor_charged_again_and_binds_to_its_map() {
    let cycle = map(vec![0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0], vec![0.0; 3]);
    let clamp = HardClamp::none(3).unwrap();
    let noise = GaussianDisturbance::try_new(vec![0.0; 3], identity(3)).unwrap();
    let mut routes = LinearRouteSet::new(&cycle);
    assert!(routes.is_empty() && routes.get(&cycle, &clamp).is_none());
    let (route, computed) = routes.route(&cycle, &clamp).unwrap();
    assert!(computed);
    let (again, computed) = routes.route(&cycle, &clamp).unwrap();
    assert!(!computed && again == route && routes.len() == 1);
    assert_eq!(routes.get(&cycle, &clamp), Some(&route));

    let compile = |model: &LinearStructuralMap, max_work, held: Option<&LinearRouteSet>| {
        compile_routed_with(
            &component_for(model),
            model,
            &noise,
            (&clamp, &[]),
            CompileBudget { max_work, max_bytes: u64::MAX },
            held,
        )
    };
    let routed = compile(&cycle, u64::MAX, None).unwrap();
    let shared = compile(&cycle, u64::MAX, Some(&routes)).unwrap();
    assert_eq!(routed.route(), shared.route());
    assert_eq!(routed.equilibrium().mean(), shared.equilibrium().mean());
    assert_eq!(routed.equilibrium().covariance(), shared.equilibrium().covariance());
    let routing = gaussian_equilibrium_route_work(3).unwrap();
    assert_eq!(
        routed.equilibrium().admission().work(),
        shared.equilibrium().admission().work() + routing,
    );
    // The held route spends only the factorization, so a budget of exactly that admits it
    // and refuses a compile that has to route.
    let factorization = shared.equilibrium().admission().work();
    assert!(compile(&cycle, factorization, Some(&routes)).is_ok());
    assert!(matches!(
        compile(&cycle, factorization, None),
        Err(SccCompileError::Budget { resource: "work", .. })
    ));

    // The set answers only for the coefficients it was built for.
    let broken = map(vec![0.0, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2, 0.0], vec![0.0; 3]);
    assert!(!routes.binds(&broken) && routes.get(&broken, &clamp).is_none());
    let other = compile(&broken, u64::MAX, Some(&routes)).unwrap();
    let alone = compile(&broken, u64::MAX, None).unwrap();
    assert_eq!(other.route(), alone.route());
    assert_ne!(other.route(), shared.route());
    assert_eq!(other.equilibrium().admission(), alone.equilibrium().admission());
    let mut rebound = routes.clone();
    let (_, computed) = rebound.route(&broken, &clamp).unwrap();
    assert!(computed && rebound.len() == routes.len());
}

#[test]
fn a_disturbance_family_checks_its_covariance_once() {
    let noise = GaussianDisturbance::try_new(vec![0.0; 2], identity(2)).unwrap();
    let shifted = noise.with_mean(vec![1.0, 2.0]).unwrap();
    assert!(!noise.psd_verified() && !shifted.psd_verified());
    shifted.require_psd_once().unwrap();
    assert!(noise.psd_verified() && noise.clone().psd_verified());
    let unrelated = GaussianDisturbance::try_new(vec![0.0; 2], identity(2)).unwrap();
    assert!(!unrelated.psd_verified());
    assert_eq!(noise, unrelated);
    assert!(noise.with_mean(vec![f64::NAN, 0.0]).is_err());
    assert!(noise.with_mean(vec![0.0]).is_err());

    let indefinite = GaussianDisturbance::try_new(vec![0.0; 2], vec![1.0, 2.0, 2.0, 1.0]).unwrap();
    assert!(matches!(
        indefinite.with_mean(vec![1.0, 0.0]).unwrap().require_psd_once(),
        Err(SccCompileError::NotPositiveSemidefinite { .. })
    ));
    assert!(!indefinite.psd_verified());
}

#[test]
fn support_route_keeps_dag_ties_and_refuses_tiny_budget_before_routing() {
    let graph = map(
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
        vec![0.0; 4],
    );
    let clamp = HardClamp::none(4).unwrap();
    assert_eq!(
        route_linear_support(&graph, &clamp).unwrap(),
        LinearSupportRoute::Acyclic { order: vec![1, 3, 2, 0] }
    );
    let noise = GaussianDisturbance::try_new(vec![0.0; 4], identity(4)).unwrap();
    let err = compile_routed_gaussian_equilibrium(
        &component_for(&graph),
        &graph,
        &noise,
        &clamp,
        &[],
        CompileBudget { max_work: 1, max_bytes: u64::MAX },
    )
    .unwrap_err();
    assert!(matches!(err, SccCompileError::Budget { resource: "work", .. }));
    let cycle = map(vec![0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0], vec![0.0; 3]);
    let clamp = HardClamp::none(3).unwrap();
    let first = route_linear_support(&cycle, &clamp).unwrap();
    assert_eq!(first, route_linear_support(&cycle, &clamp).unwrap());
    let LinearSupportRoute::Cyclic { next_attempt, blocks } = first else {
        panic!("cycle must be routed cyclic");
    };
    assert!(next_attempt.contains("edge 0->1"));
    assert!(blocks.iter().any(|block| block.len() == 3));
}

#[test]
fn acyclic_support_uses_the_dag_path_and_a_cycle_is_refused() {
    // x1 = 0.5 x0 + ε1, x2 = 0.25 x0 + 0.4 x1 + ε2, x0 = ε0. No directed cycle.
    let chain = map(vec![0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.25, 0.4, 0.0], vec![0.0, 0.0, 0.0]);
    let open = HardClamp::none(3).expect("open chain");
    let noise_mean = vec![1.0, 0.0, 0.0];
    let noise_cov = identity(3);
    let routed = compile_routed_gaussian_equilibrium(
        &component_for(&chain),
        &chain,
        &GaussianDisturbance::try_new(noise_mean.clone(), noise_cov.clone()).expect("noise"),
        &open,
        &[],
        CompileBudget::unlimited(),
    )
    .expect("acyclic route");
    match routed.route() {
        LinearSupportRoute::Acyclic { order } => assert_eq!(order, &vec![0, 1, 2]),
        LinearSupportRoute::Cyclic { next_attempt, .. } => {
            panic!("acyclic chain was refused: {next_attempt}")
        }
    }
    let law = routed.equilibrium();
    assert_eq!(law.compile_attempt(), DAG_EQUILIBRIUM_ATTEMPT);
    assert!(law.blocks().iter().all(|block| block.members.len() == 1));
    let scc = compile(&chain, noise_mean, noise_cov, &open, &[]);
    assert_eq!(scc.compile_attempt(), SCC_COMPILE_ATTEMPT);
    assert_eq!(law.mean(), scc.mean());
    assert_eq!(law.covariance(), scc.covariance());
    assert!(close(law.mean()[0], 1.0));
    assert!(close(law.mean()[1], 0.5));
    assert!(close(law.mean()[2], 0.45));

    let cycle = map(vec![0.0, 0.4, 0.5, 0.0], vec![0.0, 0.0]);
    let clamp = HardClamp::none(2).expect("open cycle");
    let routed = compile_routed_gaussian_equilibrium(
        &component_for(&cycle),
        &cycle,
        &GaussianDisturbance::try_new(vec![1.0, 0.0], identity(2)).expect("noise"),
        &clamp,
        &[],
        CompileBudget::unlimited(),
    )
    .expect("cycle is compiled only after the DAG refusal");
    let LinearSupportRoute::Cyclic { next_attempt, blocks } = routed.route() else {
        panic!("a directed cycle must be refused by the DAG algorithm");
    };
    assert!(next_attempt.contains("condensation"), "{next_attempt}");
    assert!(next_attempt.contains("sigma_separates"), "{next_attempt}");
    assert!(next_attempt.contains("next attempt"), "{next_attempt}");
    assert!(blocks.iter().any(|block| block.len() > 1), "{blocks:?}");
    assert_eq!(routed.equilibrium().compile_attempt(), SCC_COMPILE_ATTEMPT);
    assert!(routed.equilibrium().blocks().iter().any(|block| block.members.len() > 1));
}

fn identity(n: usize) -> Vec<f64> {
    let mut covariance = vec![0.0; n * n];
    for node in 0..n {
        covariance[node * n + node] = 1.0;
    }
    covariance
}

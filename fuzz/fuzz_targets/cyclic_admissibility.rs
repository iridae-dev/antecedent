//! Linear cyclic admissibility.
//!
//! A nonzero structural diagonal is refused. `k_ff` is the structural block,
//! not a stored inverse, and a forged family certificate does not admit the
//! component. A numerically singular block is not a successful solve.
#![no_main]

use antecedent_model::CyclicComponent;
use antecedent_model::cyclic_component::CyclicComponentError;
use antecedent_model::linear_solution::{
    Conditioning, FREE_BLOCK_EQUATION, FamilyCertificate, HardClamp, IdentificationStatus,
    LinearSolutionError, LinearStructuralMap, PhysicalStability, SolverConvergence, Uniqueness,
    assess,
};
use antecedent_model::scc_compile::{
    CompileBudget, GaussianDisturbance, SccCompileError, compile_gaussian_equilibrium,
};
use libfuzzer_sys::fuzz_target;

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-8
}

fn map(coefficients: Vec<f64>) -> LinearStructuralMap {
    LinearStructuralMap::try_new(coefficients, vec![0.0, 0.0]).expect("zero diagonal")
}

fn noise() -> GaussianDisturbance {
    GaussianDisturbance::try_new(vec![0.0, 0.0], vec![1.0, 0.0, 0.0, 1.0]).expect("noise")
}

fn component(model: &LinearStructuralMap) -> CyclicComponent {
    let family = assess(model, &HardClamp::none(2).expect("clamp")).expect("assess").family;
    CyclicComponent::try_new("feedback", "source-measurements", &["solve-note"], model, &family)
        .expect("admitted")
}

/// A diagonal that is not exactly zero is refused before any solve.
fn nonzero_diagonal_is_refused() {
    let next = antecedent_model::linear_solution::NONZERO_DIAGONAL_NEXT_ATTEMPT;
    let Err(error) = LinearStructuralMap::try_new(vec![0.2, 0.0, 0.0, 0.0], vec![0.0, 0.0]) else {
        panic!("nonzero diagonal was accepted");
    };
    assert_eq!(error, LinearSolutionError::NonzeroDiagonal { node: 0, next_attempt: next });
    assert!(error.to_string().contains(next), "{error}");
}

/// `k_ff` stays `K`. A certificate from another model does not admit this one,
/// and a component bound to this model does not solve a different model.
fn stored_inverse_is_not_success() {
    let coefficients = vec![0.0, 0.4, 0.5, 0.0];
    let model = map(coefficients.clone());
    let clamp = HardClamp::none(2).expect("clamp");
    let report = assess(&model, &clamp).expect("assess");
    assert_eq!(report.original.factor.equation, FREE_BLOCK_EQUATION);
    assert_eq!(report.original.factor.k_ff, coefficients);
    assert_ne!(report.original.factor.k_ff, vec![1.25, 0.5, 0.625, 1.25]);
    assert_eq!(report.solver, SolverConvergence::NotRun);
    assert_eq!(report.physical_stability, PhysicalStability::Unclaimed);
    assert_eq!(report.family.identification(), IdentificationStatus::NotMarkedUnidentified);

    let forged = FamilyCertificate::Admitted { rho_abs_upper: 0.1 };
    let mismatch = CyclicComponent::try_new(
        "feedback",
        "source-measurements",
        &["solve-note"],
        &model,
        &forged,
    );
    assert!(matches!(mismatch, Err(CyclicComponentError::CertificateModelMismatch { .. })));

    let law = compile_gaussian_equilibrium(
        &component(&model),
        &model,
        &GaussianDisturbance::try_new(vec![1.0, 0.0], vec![1.0, 0.0, 0.0, 1.0]).expect("noise"),
        &clamp,
        &[],
        CompileBudget::unlimited(),
    )
    .expect("solve");
    assert_eq!(law.free_block_equation(), FREE_BLOCK_EQUATION);
    assert_eq!(law.solver(), SolverConvergence::NotRun);
    assert_eq!(law.identification(), IdentificationStatus::NotMarkedUnidentified);
    assert!(close(law.mean()[0], 1.25));
    assert!(close(law.mean()[1], 0.625));
    assert!(close(law.covariance()[0], 1.8125));
    assert!(close(law.covariance()[1], 1.40625));
    assert!(close(law.covariance()[3], 1.953125));
    assert!(law.residual_inf() < 1e-8);

    let other = map(vec![0.0, 1.0, 1.0, 0.0]);
    let reused = compile_gaussian_equilibrium(
        &component(&model),
        &other,
        &noise(),
        &HardClamp::none(2).expect("clamp"),
        &[],
        CompileBudget::unlimited(),
    );
    assert!(matches!(reused, Err(SccCompileError::ComponentModelMismatch { .. })));
}

/// A pivot under the floor is numerically singular. That is not a unique solve
/// and not an exact singularity, and it does not admit a component.
fn numerically_singular_block_is_not_a_successful_solve() {
    let near = map(vec![0.0, 1.0, 1.0 + 1.0e-13, 0.0]);
    let report = assess(&near, &HardClamp::none(2).expect("clamp")).expect("assess");
    assert_eq!(report.original.uniqueness, Uniqueness::NumericallySingular { members: vec![0, 1] });
    assert_eq!(report.original.conditioning, Conditioning::Undefined);
    assert!(!matches!(report.original.uniqueness, Uniqueness::Unique));
    assert!(!matches!(report.original.uniqueness, Uniqueness::Singular(_)));
    assert_eq!(report.solver, SolverConvergence::NotRun);
    let refused = CyclicComponent::try_new(
        "feedback",
        "source-measurements",
        &["solve-note"],
        &near,
        &report.family,
    );
    assert!(matches!(refused, Err(CyclicComponentError::FamilyNotAdmitted { .. })));

    let exact = map(vec![0.0, 1.0, 1.0, 0.0]);
    let exact_report = assess(&exact, &HardClamp::none(2).expect("clamp")).expect("assess");
    assert!(matches!(exact_report.original.uniqueness, Uniqueness::Singular(_)));
    assert!(!exact_report.family.is_admitted());
}

fn coefficient(byte: u8) -> f64 {
    f64::from(byte) / 64.0 - 2.0
}

fuzz_target!(|data: &[u8]| {
    nonzero_diagonal_is_refused();
    stored_inverse_is_not_success();
    numerically_singular_block_is_not_a_successful_solve();
    if data.len() < 2 {
        return;
    }
    let coefficients = vec![0.0, coefficient(data[0]), coefficient(data[1]), 0.0];
    let mut poked = coefficients.clone();
    poked[0] = if data[0] == 0 { 0.2 } else { f64::from(data[0]) };
    let diagonal = LinearStructuralMap::try_new(poked, vec![0.0, 0.0]);
    assert!(matches!(diagonal, Err(LinearSolutionError::NonzeroDiagonal { node: 0, .. })));

    let model = map(coefficients.clone());
    let report = assess(&model, &HardClamp::none(2).expect("clamp")).expect("assess");
    assert_eq!(report.original.factor.k_ff, coefficients);
    assert_eq!(report.original.factor.equation, FREE_BLOCK_EQUATION);
    assert_eq!(report.original.factor.k_ff[0].to_bits(), 0.0_f64.to_bits());
    assert_eq!(report.original.factor.k_ff[3].to_bits(), 0.0_f64.to_bits());
    assert_eq!(report.solver, SolverConvergence::NotRun);
    assert_eq!(report.physical_stability, PhysicalStability::Unclaimed);
    assert_eq!(report.family.identification(), IdentificationStatus::NotMarkedUnidentified);

    let det = 1.0 - coefficients[1] * coefficients[2];
    if det.to_bits() == 0.0_f64.to_bits() {
        assert!(matches!(report.original.uniqueness, Uniqueness::Singular(_)));
    }
    if matches!(report.original.uniqueness, Uniqueness::NumericallySingular { .. }) {
        assert_eq!(report.original.conditioning, Conditioning::Undefined);
        let refused = CyclicComponent::try_new(
            "feedback",
            "source-measurements",
            &["solve-note"],
            &model,
            &report.family,
        );
        assert!(matches!(refused, Err(CyclicComponentError::FamilyNotAdmitted { .. })));
    }
    if report.family.is_admitted() {
        let law = compile_gaussian_equilibrium(
            &component(&model),
            &model,
            &noise(),
            &HardClamp::none(2).expect("clamp"),
            &[],
            CompileBudget::unlimited(),
        )
        .expect("admitted block solves");
        assert_eq!(law.solver(), SolverConvergence::NotRun);
        assert_eq!(law.free_block_equation(), FREE_BLOCK_EQUATION);
        assert!(law.mean().iter().all(|value| value.abs() <= 1e-8));
        assert!(law.residual_inf() <= 1e-8);
    }
});

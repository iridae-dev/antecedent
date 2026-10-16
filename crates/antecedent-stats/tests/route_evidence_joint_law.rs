//! Joint-law arithmetic: one assertion set per public routine, each against its
//! own hand-derived value.
//!
//! Oracles. For binary marginals `X ~ Bernoulli(p)`, `Y ~ Bernoulli(q)` with
//! `p = 0.4`, `q = 0.7`, every coupling is fixed by `c = P(X = 1, Y = 1)`,
//! whose Frechet range is `[max(0, p + q - 1), min(p, q)] = [0.1, 0.4]`; the
//! other three cells follow: `P(1, 0) = p - c`, `P(0, 1) = q - c` and
//! `P(0, 0) = 1 - p - q + c`. Every envelope below is read off that one
//! parameter, never from the library.

use antecedent_core::CancellationToken;
use antecedent_stats::{
    JointLawError, Marginal, MomentSummary, PairStructure, ValueFunctional,
    aligned_functional_mean, correlation_range_envelope, covariance_envelope,
    difference_cdf_envelope, frechet_expectation_envelope, functional_envelope,
    linear_combination_moments, linear_variance_envelope, mixture_moments, paired_moments,
    product_moment_envelope, sum_cdf_envelope,
};

const TOL: f64 = 1e-12;

fn token() -> CancellationToken {
    CancellationToken::new()
}

fn near(got: f64, want: f64) {
    assert!((got - want).abs() < TOL, "{got} vs {want}");
}

fn bernoulli(p: f64) -> Marginal {
    Marginal::from_weighted(&[0.0, 1.0], &[1.0 - p, p]).expect("bernoulli marginal")
}

#[test]
fn the_sum_cdf_envelope_is_the_makarov_range_of_the_cell_that_decides_it() {
    let (x, y) = (bernoulli(0.4), bernoulli(0.7));
    // P(X + Y <= 0) = P(0, 0) = 1 - p - q + c = c - 0.1, c in [0.1, 0.4].
    let at0 = sum_cdf_envelope(&x, &y, 0.0).expect("t = 0");
    near(at0.lower, 0.0);
    near(at0.upper, 0.3);
    // P(X + Y <= 1) = 1 - c.
    let at1 = sum_cdf_envelope(&x, &y, 1.0).expect("t = 1");
    near(at1.lower, 0.6);
    near(at1.upper, 0.9);
    // Below the support the probability is zero for every coupling.
    let below = sum_cdf_envelope(&x, &y, -0.5).expect("t < support");
    assert!(below.is_degenerate());
    near(below.lower, 0.0);
    // A non-finite threshold is refused, never clamped.
    assert!(matches!(
        sum_cdf_envelope(&x, &y, f64::NAN),
        Err(JointLawError::NonFinite("threshold"))
    ));
}

#[test]
fn the_difference_cdf_envelope_negates_the_second_marginal() {
    let (x, y) = (bernoulli(0.4), bernoulli(0.7));
    // P(X - Y <= 0) = 1 - P(1, 0) with P(1, 0) = p - c in [0, 0.3].
    let at0 = difference_cdf_envelope(&x, &y, 0.0).expect("t = 0");
    near(at0.lower, 0.7);
    near(at0.upper, 1.0);
    // P(X - Y <= -1) = P(0, 1) = q - c in [0.3, 0.6].
    let at_minus_one = difference_cdf_envelope(&x, &y, -1.0).expect("t = -1");
    near(at_minus_one.lower, 0.3);
    near(at_minus_one.upper, 0.6);
    // X - Y never exceeds one.
    let top = difference_cdf_envelope(&x, &y, 1.0).expect("t = 1");
    assert!(top.is_degenerate());
    near(top.lower, 1.0);
}

#[test]
fn the_product_moment_envelope_is_attained_at_the_monotone_couplings() {
    // Binary: E[XY] = c in [0.1, 0.4].
    let binary = product_moment_envelope(&bernoulli(0.4), &bernoulli(0.7), &token())
        .expect("binary product");
    near(binary.lower, 0.1);
    near(binary.upper, 0.4);
    // X uniform on {1, 2}, Y uniform on {1, 3}: comonotone pairs (1,1),(2,3)
    // give (1 + 6) / 2 = 3.5, countermonotone (1,3),(2,1) give (3 + 2) / 2 = 2.5.
    let x = Marginal::from_draws(&[1.0, 2.0]).expect("x");
    let y = Marginal::from_draws(&[1.0, 3.0]).expect("y");
    let two = product_moment_envelope(&x, &y, &token()).expect("two-atom product");
    near(two.lower, 2.5);
    near(two.upper, 3.5);
}

#[test]
fn the_covariance_envelope_subtracts_the_product_of_the_means() {
    // Cov = c - p q = c - 0.28 with c in [0.1, 0.4].
    let cov = covariance_envelope(&bernoulli(0.4), &bernoulli(0.7), &token()).expect("cov");
    near(cov.lower, -0.18);
    near(cov.upper, 0.12);
    // The range always contains the independent coupling's zero.
    assert!(cov.lower < 0.0 && cov.upper > 0.0);
}

#[test]
fn the_linear_variance_envelope_is_the_covariance_range_through_the_cross_term() {
    let (x, y) = (bernoulli(0.4), bernoulli(0.7));
    // Var X = 0.24, Var Y = 0.21, so a^2 Var X + b^2 Var Y = 1.17 for (2, 1).
    // The cross term 2ab = 4 multiplies Cov in [-0.18, 0.12].
    let sum = linear_variance_envelope(2.0, 1.0, &x, &y, &token()).expect("2X + Y");
    near(sum.lower, 1.17 - 0.72);
    near(sum.upper, 1.17 + 0.48);
    // (2, -1): the cross term is -4, so the roles of the endpoints swap.
    let difference = linear_variance_envelope(2.0, -1.0, &x, &y, &token()).expect("2X - Y");
    near(difference.lower, 1.17 - 0.48);
    near(difference.upper, 1.17 + 0.72);
    // A non-finite coefficient is refused.
    assert!(matches!(
        linear_variance_envelope(f64::INFINITY, 1.0, &x, &y, &token()),
        Err(JointLawError::NonFinite(_))
    ));
}

#[test]
fn a_user_value_function_gets_exact_coupling_extremes_only_with_its_sign_verified() {
    let (x, y) = (bernoulli(0.4), bernoulli(0.7));
    // E[(X - Y)^2] = P(X != Y) = p + q - 2c in [1.1 - 0.8, 1.1 - 0.2] = [0.3, 0.9];
    // (x - y)^2 is submodular, so the extremes sit at the monotone couplings.
    let squared = frechet_expectation_envelope(
        &x,
        &y,
        &|a, b| (a - b) * (a - b),
        PairStructure::Submodular,
        &token(),
    )
    .expect("submodular value");
    near(squared.lower, 0.3);
    near(squared.upper, 0.9);
    // Declared supermodular, the same function fails the lattice check.
    assert!(matches!(
        frechet_expectation_envelope(
            &x,
            &y,
            &|a, b| (a - b) * (a - b),
            PairStructure::Supermodular,
            &token(),
        ),
        Err(JointLawError::StructureNotVerified { structure: "supermodular" })
    ));
}

#[test]
fn a_named_functional_envelope_uses_its_own_declared_sign_structure() {
    let (x, y) = (bernoulli(0.4), bernoulli(0.7));
    // E[max(X, Y)] = P(X = 1 or Y = 1) = p + q - c in [0.7, 1.0].
    let maximum = functional_envelope(&x, &y, ValueFunctional::Maximum, &token()).expect("max");
    near(maximum.lower, 0.7);
    near(maximum.upper, 1.0);
    // E[min(X, Y)] = c in [0.1, 0.4].
    let minimum = functional_envelope(&x, &y, ValueFunctional::Minimum, &token()).expect("min");
    near(minimum.lower, 0.1);
    near(minimum.upper, 0.4);
    assert_eq!(ValueFunctional::Maximum.structure(), PairStructure::Submodular);
    assert_eq!(ValueFunctional::Minimum.structure(), PairStructure::Supermodular);
}

#[test]
fn a_declared_correlation_range_is_linear_in_the_correlation() {
    let x = MomentSummary { mean: 2.0, standard_deviation: 3.0 };
    let y = MomentSummary { mean: -1.0, standard_deviation: 2.0 };
    // scale s_x s_y = 6, centre mu_x mu_y = -2, base 4 * 9 + 1 * 4 = 40,
    // Var(2X + Y) = 40 + 24 rho, over rho in [0, 0.5].
    let range = correlation_range_envelope(x, y, 2.0, 1.0, 0.0, 0.5).expect("range");
    near(range.product_moment.lower, -2.0);
    near(range.product_moment.upper, 1.0);
    near(range.covariance.lower, 0.0);
    near(range.covariance.upper, 3.0);
    near(range.linear_variance.lower, 40.0);
    near(range.linear_variance.upper, 52.0);
    // A single correlation is a covariance declaration, not a range.
    assert!(matches!(
        correlation_range_envelope(x, y, 2.0, 1.0, 0.25, 0.25),
        Err(JointLawError::InvalidRange(_))
    ));
    // A range outside [-1, 1] and a negative standard deviation are refused.
    assert!(correlation_range_envelope(x, y, 2.0, 1.0, -1.5, 0.5).is_err());
    let negative = MomentSummary { mean: 0.0, standard_deviation: -1.0 };
    assert!(matches!(
        correlation_range_envelope(negative, y, 1.0, 1.0, 0.0, 0.5),
        Err(JointLawError::InvalidVariance(_))
    ));
}

#[test]
fn linear_combination_moments_use_the_whole_covariance_matrix() {
    // a = (2, -1), mu = (1, 3): mean 2 - 3 = -1.
    // a' S a = 4 * 1 + 1 * 4 + 2 * (2)(-1)(0.5) = 6.
    let moments = linear_combination_moments(&[2.0, -1.0], &[1.0, 3.0], &[1.0, 0.5, 0.5, 4.0])
        .expect("moments");
    near(moments.mean, -1.0);
    near(moments.variance, 6.0);
    // A diagonal matrix drops the cross term: 4 + 4 = 8.
    let diagonal = linear_combination_moments(&[2.0, -1.0], &[1.0, 3.0], &[1.0, 0.0, 0.0, 4.0])
        .expect("diagonal");
    near(diagonal.variance, 8.0);
    // An asymmetric matrix and a shape mismatch are refused.
    assert!(matches!(
        linear_combination_moments(&[1.0, 1.0], &[0.0, 0.0], &[1.0, 0.9, 0.1, 1.0]),
        Err(JointLawError::Covariance(_))
    ));
    assert!(matches!(
        linear_combination_moments(&[1.0, 1.0], &[0.0], &[1.0, 0.0, 0.0, 1.0]),
        Err(JointLawError::Shape(_))
    ));
}

#[test]
fn mixture_moments_split_the_total_variance_into_within_and_between() {
    // Weights (1/2, 1/2), means (0, 2), variances (1, 3):
    // mean 1, within 2, between 1/2 * 1 + 1/2 * 1 = 1, total 3.
    let mixture = mixture_moments(&[0.5, 0.5], &[0.0, 2.0], &[1.0, 3.0]).expect("mixture");
    near(mixture.mean, 1.0);
    near(mixture.within, 2.0);
    near(mixture.between, 1.0);
    near(mixture.variance, 3.0);
    // Weights that do not sum to one and a negative variance are refused.
    assert!(mixture_moments(&[0.5, 0.4], &[0.0, 2.0], &[1.0, 3.0]).is_err());
    assert!(mixture_moments(&[0.5, 0.5], &[0.0, 2.0], &[1.0, -3.0]).is_err());
}

#[test]
fn paired_moments_are_population_moments_of_the_aligned_rows() {
    // x = 0..3, y = (1, 3, 2, 6): means 1.5 and 3; var x = 1.25;
    // y deviations (-2, 0, -1, 3) give var y = 14 / 4 = 3.5; cov = 7 / 4.
    let paired = paired_moments(&[0.0, 1.0, 2.0, 3.0], &[1.0, 3.0, 2.0, 6.0]).expect("paired");
    assert_eq!(paired.n, 4);
    near(paired.mean_x, 1.5);
    near(paired.mean_y, 3.0);
    near(paired.var_x, 1.25);
    near(paired.var_y, 3.5);
    near(paired.covariance, 1.75);
    // Unequal and empty columns are refused.
    assert!(matches!(paired_moments(&[1.0, 2.0], &[1.0]), Err(JointLawError::Shape(_))));
    assert!(matches!(paired_moments(&[], &[]), Err(JointLawError::Shape(_))));
}

#[test]
fn an_aligned_functional_mean_is_the_row_average_of_the_functional() {
    let x = [0.0, 1.0, 2.0, 3.0];
    let y = [1.0, 3.0, 2.0, 6.0];
    // x * y = (0, 3, 4, 18): mean 25 / 4.
    near(
        aligned_functional_mean(&x, &y, ValueFunctional::Product, &token()).expect("product"),
        6.25,
    );
    // min(x, y) = (0, 1, 2, 3): mean 1.5.
    near(
        aligned_functional_mean(&x, &y, ValueFunctional::Minimum, &token()).expect("minimum"),
        1.5,
    );
    // (x - y)^2 = (1, 4, 0, 9): mean 3.5.
    near(
        aligned_functional_mean(&x, &y, ValueFunctional::SquaredDifference, &token())
            .expect("squared difference"),
        3.5,
    );
    // A ratio with a zero denominator is refused, not averaged as infinity.
    assert!(matches!(
        aligned_functional_mean(&[1.0, 1.0], &[1.0, 0.0], ValueFunctional::Ratio, &token()),
        Err(JointLawError::NonFiniteValue)
    ));
}

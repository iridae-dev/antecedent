use crate::quantile::QuantileRule;
use antecedent_core::CausalRng;

use super::*;

fn token() -> CancellationToken {
    CancellationToken::new()
}

fn normal_pair(seed: u64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = CausalRng::from_seed(seed);
    let mut z = Vec::with_capacity(n);
    while z.len() < 2 * n {
        let u1 = 1.0 - rng.next_f64();
        let u2 = rng.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f64::consts::PI * u2;
        z.push(r * t.cos());
        z.push(r * t.sin());
    }
    let (z1, z2): (Vec<f64>, Vec<f64>) = z.chunks(2).map(|c| (c[0], c[1])).unzip();
    (z1, z2[..n].to_vec())
}

#[test]
fn linear_variance_of_a_contrast_is_the_closed_form() {
    // Var(X - Y) = sx^2 + sy^2 - 2 cov, with sx^2 = 4, sy^2 = 9, cov = 3.
    let covariance = [4.0, 3.0, 3.0, 9.0];
    let moments =
        linear_combination_moments(&[1.0, -1.0], &[5.0, 2.0], &covariance).expect("moments");
    assert!((moments.mean - 3.0).abs() < 1e-15);
    assert!((moments.variance - (4.0 + 9.0 - 6.0)).abs() < 1e-12);
    // Weighted sum 0.25 X + 0.75 Y.
    let w = linear_combination_moments(&[0.25, 0.75], &[5.0, 2.0], &covariance).expect("w");
    let expected = 0.0625 * 4.0 + 0.5625 * 9.0 + 2.0 * 0.25 * 0.75 * 3.0;
    assert!((w.variance - expected).abs() < 1e-12);
    assert!((w.mean - (1.25 + 1.5)).abs() < 1e-15);
}

#[test]
fn a_covariance_that_is_not_positive_semidefinite_is_refused() {
    let covariance = [1.0, 2.0, 2.0, 1.0];
    let error =
        linear_combination_moments(&[1.0, 1.0], &[0.0, 0.0], &covariance).expect_err("refuse");
    assert!(matches!(
        error,
        JointLawError::Covariance(SynthesisError::NotPositiveSemidefinite { .. })
    ));
}

#[test]
fn known_weight_mixture_matches_the_closed_form() {
    // Components N(0, 1) and N(4, 4) with weights 0.25 and 0.75.
    let m = mixture_moments(&[0.25, 0.75], &[0.0, 4.0], &[1.0, 4.0]).expect("mixture");
    assert!((m.mean - 3.0).abs() < 1e-15);
    assert!((m.within - (0.25 + 3.0)).abs() < 1e-15);
    assert!((m.between - (0.25 * 9.0 + 0.75 * 1.0)).abs() < 1e-14);
    // Second moment route: E[X^2] = 0.25 (1) + 0.75 (4 + 16) = 15.25.
    assert!((m.variance - (15.25 - 9.0)).abs() < 1e-13);
    assert!(matches!(
        mixture_moments(&[0.5, 0.4], &[0.0, 1.0], &[1.0, 1.0]),
        Err(JointLawError::InvalidWeights(_))
    ));
}

#[test]
fn aligned_draws_recover_the_product_moment_and_unpaired_draws_do_not() {
    let (mu_x, mu_y, s_x, s_y, rho) = (1.0, 2.0, 1.5, 0.5, 0.6);
    let n = 200_000;
    let (z1, z2) = normal_pair(20_240_917, n);
    let x: Vec<f64> = z1.iter().map(|z| mu_x + s_x * z).collect();
    let y: Vec<f64> = z1
        .iter()
        .zip(&z2)
        .map(|(a, b)| mu_y + s_y * (rho * a + (1.0 - rho * rho).sqrt() * b))
        .collect();
    let truth = rho * s_x * s_y + mu_x * mu_y;
    let var = mu_x * mu_x * s_y * s_y
        + mu_y * mu_y * s_x * s_x
        + 2.0 * mu_x * mu_y * rho * s_x * s_y
        + s_x * s_x * s_y * s_y * (1.0 + rho * rho);
    let band = 5.0 * (var / n as f64).sqrt();
    let aligned =
        aligned_functional_mean(&x, &y, ValueFunctional::Product, &token()).expect("mean");
    assert!((aligned - truth).abs() < band, "aligned {aligned} truth {truth} band {band}");
    // The same columns shuffled apart by an independent permutation behave
    // as unrelated marginals: the product moment is mu_x mu_y, outside the
    // band of the joint truth.
    let shifted: Vec<f64> = y.iter().cycle().skip(7_919).take(n).copied().collect();
    let paired_by_index =
        aligned_functional_mean(&x, &shifted, ValueFunctional::Product, &token()).expect("mean");
    assert!((paired_by_index - truth).abs() > 5.0 * band);
    let moments = paired_moments(&x, &y).expect("moments");
    assert!((moments.covariance - rho * s_x * s_y).abs() < 5.0 * s_x * s_y / (n as f64).sqrt());
}

#[test]
fn exact_finite_joint_law_enumeration_of_a_nonlinear_functional() {
    // Joint law over (X, Y): (0,0) 0.1, (0,2) 0.4, (1,0) 0.3, (1,2) 0.2.
    let x = [0.0, 0.0, 1.0, 1.0];
    let y = [0.0, 2.0, 0.0, 2.0];
    // Replicate rows by mass: ten rows with the stated frequencies.
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for (i, count) in [1, 4, 3, 2].into_iter().enumerate() {
        for _ in 0..count {
            xs.push(x[i]);
            ys.push(y[i]);
        }
    }
    let product =
        aligned_functional_mean(&xs, &ys, ValueFunctional::Product, &token()).expect("product");
    assert!((product - 0.4).abs() < 1e-15, "E[XY] = 1 * 2 * 0.2");
    let minimum =
        aligned_functional_mean(&xs, &ys, ValueFunctional::Minimum, &token()).expect("minimum");
    assert!((minimum - 0.2).abs() < 1e-15);
    let squared = aligned_functional_mean(&xs, &ys, ValueFunctional::SquaredDifference, &token())
        .expect("squared");
    // 0.1*0 + 0.4*4 + 0.3*1 + 0.2*1 = 2.1
    assert!((squared - 2.1).abs() < 1e-14);
}

#[test]
fn bernoulli_envelopes_equal_the_closed_form_endpoints() {
    let (p, q) = (0.3, 0.6);
    let x = Marginal::from_weighted(&[0.0, 1.0], &[1.0 - p, p]).expect("x");
    let y = Marginal::from_weighted(&[0.0, 1.0], &[1.0 - q, q]).expect("y");
    let c = token();
    let product = product_moment_envelope(&x, &y, &c).expect("product");
    // P(1,1) in [max(0, p+q-1), min(p, q)] = [0, 0.3].
    assert!((product.lower - 0.0).abs() < 1e-12);
    assert!((product.upper - 0.3).abs() < 1e-12);
    let cov = covariance_envelope(&x, &y, &c).expect("cov");
    assert!((cov.lower + 0.18).abs() < 1e-12 && (cov.upper - 0.12).abs() < 1e-12);
    let var_sum = linear_variance_envelope(1.0, 1.0, &x, &y, &c).expect("var");
    assert!((var_sum.lower - 0.09).abs() < 1e-12 && (var_sum.upper - 0.69).abs() < 1e-12);
    let var_diff = linear_variance_envelope(1.0, -1.0, &x, &y, &c).expect("var");
    assert!((var_diff.lower - 0.21).abs() < 1e-12 && (var_diff.upper - 0.81).abs() < 1e-12);
    // P(X+Y <= 0) = 1 - p - q + c in [0.1, 0.4]; P(X+Y <= 1) = 1 - c in [0.7, 1].
    let at0 = sum_cdf_envelope(&x, &y, 0.0).expect("cdf");
    assert!((at0.lower - 0.1).abs() < 1e-12 && (at0.upper - 0.4).abs() < 1e-12);
    let at1 = sum_cdf_envelope(&x, &y, 1.0).expect("cdf");
    assert!((at1.lower - 0.7).abs() < 1e-12 && (at1.upper - 1.0).abs() < 1e-12);
    let at2 = sum_cdf_envelope(&x, &y, 2.0).expect("cdf");
    assert!(at2.is_degenerate() && (at2.lower - 1.0).abs() < 1e-15);
}

#[test]
fn coupling_envelope_keeps_atoms_below_the_mass_tolerance() {
    let tiny = 1e-13;
    let x = Marginal::from_weighted(&[0.0, 10.0], &[1.0 - tiny, tiny]).expect("x");
    let y = Marginal::from_weighted(&[0.0, 1.0], &[1.0 - tiny, tiny]).expect("y");
    let product = product_moment_envelope(&x, &y, &token()).expect("product envelope");
    // The comonotone coupling pairs the two rare upper atoms, contributing
    // 10 * tiny. The old residual-mass cutoff silently discarded this cell.
    assert!((product.upper - 10.0 * tiny).abs() < 1e-25, "{product:?}");
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn go(current: &mut Vec<usize>, used: &mut [bool], out: &mut Vec<Vec<usize>>) {
        if current.len() == used.len() {
            out.push(current.clone());
            return;
        }
        for i in 0..used.len() {
            if !used[i] {
                used[i] = true;
                current.push(i);
                go(current, used, out);
                current.pop();
                used[i] = false;
            }
        }
    }
    let mut out = Vec::new();
    go(&mut Vec::new(), &mut vec![false; n], &mut out);
    out
}

#[test]
fn equal_mass_marginals_envelopes_equal_permutation_enumeration() {
    // Uniform marginals on m atoms: the extreme couplings are permutations.
    let xs = [0.0, 0.25, 0.5, 0.75, 1.0];
    let ys = [0.0, 0.125, 0.5, 0.625, 1.5];
    let m = xs.len();
    let x = Marginal::from_draws(&xs).expect("x");
    let y = Marginal::from_draws(&ys).expect("y");
    let perms = permutations(m);
    let mut prod = (f64::INFINITY, f64::NEG_INFINITY);
    for sigma in &perms {
        let e: f64 = (0..m).map(|i| xs[i] * ys[sigma[i]]).sum::<f64>() / m as f64;
        prod = (prod.0.min(e), prod.1.max(e));
    }
    let env = product_moment_envelope(&x, &y, &token()).expect("product");
    assert!((env.lower - prod.0).abs() < 1e-14 && (env.upper - prod.1).abs() < 1e-14);
    for t in [0.5, 0.75, 1.0, 1.125, 1.5, 2.0, 2.25] {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for sigma in &perms {
            let hits = (0..m).filter(|&i| xs[i] + ys[sigma[i]] <= t).count();
            let p = hits as f64 / m as f64;
            lo = lo.min(p);
            hi = hi.max(p);
        }
        let bounds = sum_cdf_envelope(&x, &y, t).expect("cdf");
        assert!(
            (bounds.lower - lo).abs() < 1e-12 && (bounds.upper - hi).abs() < 1e-12,
            "t={t}: envelope {bounds:?} vs enumeration ({lo}, {hi})"
        );
    }
    // Difference route against the same enumeration.
    for t in [-0.5, 0.0, 0.25] {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for sigma in &perms {
            let hits = (0..m).filter(|&i| xs[i] - ys[sigma[i]] <= t).count();
            let p = hits as f64 / m as f64;
            lo = lo.min(p);
            hi = hi.max(p);
        }
        let bounds = difference_cdf_envelope(&x, &y, t).expect("cdf");
        assert!((bounds.lower - lo).abs() < 1e-12 && (bounds.upper - hi).abs() < 1e-12);
    }
}

#[test]
fn uniform_marginals_makarov_bounds_match_the_continuous_closed_form() {
    // X, Y ~ U(0, 1) as m equal atoms (j + 0.5)/m: P(X+Y <= t) lies in
    // [max(0, t - 1), min(t, 1)] for the continuous law; the atom bounds
    // converge to it at rate 1/m.
    let m = 2000;
    let atoms: Vec<f64> = (0..m).map(|j| (f64::from(j) + 0.5) / f64::from(m)).collect();
    let u = Marginal::from_draws(&atoms).expect("u");
    for t in [0.25, 0.5, 0.9, 1.0, 1.4, 1.75] {
        let b = sum_cdf_envelope(&u, &u, t).expect("cdf");
        let (lo, hi) = ((t - 1.0_f64).max(0.0), t.min(1.0));
        assert!((b.lower - lo).abs() < 2.0 / f64::from(m), "t={t} lower {b:?}");
        assert!((b.upper - hi).abs() < 2.0 / f64::from(m), "t={t} upper {b:?}");
    }
    // E[XY] for uniform marginals: comonotone 1/3, countermonotone 1/6.
    let p = product_moment_envelope(&u, &u, &token()).expect("product");
    assert!((p.upper - 1.0 / 3.0).abs() < 1e-3 && (p.lower - 1.0 / 6.0).abs() < 1e-3);
}

#[test]
fn envelopes_are_invariant_to_listing_order_and_relabelling() {
    let a = [0.3, 1.2, -0.4, 2.0, 0.9, 0.9];
    let mut a_rev = a;
    a_rev.reverse();
    let b = [1.0, 0.1, 0.1, 4.0, -2.0, 0.5];
    let mut b_perm = b;
    b_perm.rotate_left(2);
    let (x1, x2) = (Marginal::from_draws(&a).expect("x"), Marginal::from_draws(&a_rev).expect("x"));
    let (y1, y2) =
        (Marginal::from_draws(&b).expect("y"), Marginal::from_draws(&b_perm).expect("y"));
    assert_eq!(x1, x2);
    assert_eq!(y1, y2);
    assert_eq!(
        sum_cdf_envelope(&x1, &y1, 1.0).expect("a"),
        sum_cdf_envelope(&x2, &y2, 1.0).expect("b")
    );
    // Swapping the two variables leaves the sum and product envelopes.
    assert_eq!(
        sum_cdf_envelope(&x1, &y1, 1.0).expect("a"),
        sum_cdf_envelope(&y1, &x1, 1.0).expect("b")
    );
    assert_eq!(
        product_moment_envelope(&x1, &y1, &token()).expect("a"),
        product_moment_envelope(&y1, &x1, &token()).expect("b")
    );
}

#[test]
fn a_value_function_without_the_declared_structure_is_refused() {
    let x = Marginal::from_draws(&[0.0, 1.0, 2.0]).expect("x");
    let y = Marginal::from_draws(&[0.0, 1.0, 2.0]).expect("y");
    // sin(x y) is neither super- nor submodular on this lattice.
    let error = frechet_expectation_envelope(
        &x,
        &y,
        &|a, b| (3.0 * a * b).sin(),
        PairStructure::Supermodular,
        &token(),
    )
    .expect_err("refuse");
    assert!(matches!(error, JointLawError::StructureNotVerified { .. }));
    // A submodular functional is bounded with the roles swapped.
    let env = functional_envelope(&x, &y, ValueFunctional::SquaredDifference, &token())
        .expect("envelope");
    // Comonotone coupling gives E[(X-Y)^2] = 0; countermonotone
    // pairs (0,2), (1,1), (2,0): (4 + 0 + 4)/3.
    assert!((env.lower - 0.0).abs() < 1e-15 && (env.upper - 8.0 / 3.0).abs() < 1e-14);
}

#[test]
fn correlation_range_endpoints_are_the_closed_form() {
    let x = MomentSummary { mean: 1.0, standard_deviation: 2.0 };
    let y = MomentSummary { mean: 3.0, standard_deviation: 0.5 };
    let r = correlation_range_envelope(x, y, 1.0, -1.0, -0.5, 0.5).expect("range");
    assert!((r.product_moment.lower - (3.0 - 0.5)).abs() < 1e-15);
    assert!((r.product_moment.upper - (3.0 + 0.5)).abs() < 1e-15);
    // Var(X - Y) = 4 + 0.25 - 2 rho 1.
    assert!((r.linear_variance.lower - (4.25 - 1.0)).abs() < 1e-15);
    assert!((r.linear_variance.upper - (4.25 + 1.0)).abs() < 1e-15);
    assert!(matches!(
        correlation_range_envelope(x, y, 1.0, 1.0, 0.2, 0.2),
        Err(JointLawError::InvalidRange(_))
    ));
}

#[test]
fn finite_inputs_that_overflow_declared_moment_bounds_are_refused() {
    let x = MomentSummary { mean: 1e308, standard_deviation: 0.0 };
    let y = MomentSummary { mean: 1e308, standard_deviation: 0.0 };
    assert_eq!(
        correlation_range_envelope(x, y, 1.0, 0.0, -0.5, 0.5),
        Err(JointLawError::NonFiniteValue)
    );

    let x = Marginal::from_draws(&[0.0, 1.0]).expect("x");
    let y = Marginal::from_draws(&[0.0, 1.0]).expect("y");
    assert_eq!(
        linear_variance_envelope(1e308, 1.0, &x, &y, &token()),
        Err(JointLawError::NonFiniteValue)
    );
}

#[test]
fn composed_linear_draws_have_the_closed_form_spread() {
    // 2 X - Y over aligned rows: (2 * 1 - 4, 2 * 2 - 6, 2 * 3 - 8) = -2 each.
    let x = [1.0, 2.0, 3.0];
    let y = [4.0, 6.0, 8.0];
    let rows = aligned_linear_draws(&[2.0, -1.0], &[&x, &y]).expect("rows");
    assert_eq!(rows, vec![-2.0, -2.0, -2.0]);
    let spread = draw_spread(&rows, None).expect("spread");
    assert_eq!((spread.mean, spread.variance), (-2.0, 0.0));
    // X + Y rows 5, 8, 11: mean 8, sample variance 9, se 3 / sqrt(3).
    let sum = aligned_linear_draws(&[1.0, 1.0], &[&x, &y]).expect("sum");
    let spread = draw_spread(&sum, None).expect("spread");
    assert!((spread.mean - 8.0).abs() < 1e-15 && (spread.variance - 9.0).abs() < 1e-14);
    assert!((spread.mean_standard_error - 3.0 / 3.0_f64.sqrt()).abs() < 1e-14);
    // An effective size below n widens the error: se = 3 / sqrt(1.5).
    let thin = draw_spread(&sum, Some(1.5)).expect("spread");
    assert!((thin.mean_standard_error - 3.0 / 1.5_f64.sqrt()).abs() < 1e-14);
    // The aligned variance equals the delta-method variance under the
    // columns' own covariance: Var X = 1, Var Y = 4, Cov = 2.
    let moments = linear_combination_moments(&[1.0, 1.0], &[2.0, 6.0], &[1.0, 2.0, 2.0, 4.0])
        .expect("moments");
    assert!((moments.variance - 9.0).abs() < 1e-14);
}

#[test]
fn composed_draw_refusals_name_the_shape_problem() {
    let x = [1.0, 2.0];
    let y = [1.0, 2.0, 3.0];
    let (xs, ys): (&[f64], &[f64]) = (&x, &y);
    assert!(matches!(aligned_linear_draws(&[1.0, 1.0], &[xs, ys]), Err(JointLawError::Shape(_))));
    assert!(matches!(aligned_linear_draws(&[1.0], &[xs, ys]), Err(JointLawError::Shape(_))));
    assert!(matches!(
        aligned_linear_draws(&[1.0, f64::NAN], &[xs, xs]),
        Err(JointLawError::NonFinite(_))
    ));
    assert!(matches!(draw_spread(&[1.0], None), Err(JointLawError::Shape(_))));
    assert!(matches!(draw_spread(&[1.0, 2.0], Some(3.0)), Err(JointLawError::InvalidLevel(_))));
    let zeros = [0.0, 0.0];
    assert_eq!(
        aligned_functional_draws(&x, &zeros, ValueFunctional::Ratio, &token()),
        Err(JointLawError::NonFiniteValue)
    );
    assert_eq!(
        aligned_map_draws(&[xs, ys], &|r| r[0] * r[1], &token()),
        Err(JointLawError::Shape("aligned columns differ in length"))
    );
}

#[test]
fn composed_nonlinear_rows_match_the_enumerated_joint_law() {
    // Same ten-row joint law as the functional mean test: the rows of the
    // product average to E[XY] = 0.4, and the spread is the closed form.
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for (x, y, count) in [(0.0, 0.0, 1), (0.0, 2.0, 4), (1.0, 0.0, 3), (1.0, 2.0, 2)] {
        for _ in 0..count {
            xs.push(x);
            ys.push(y);
        }
    }
    let rows =
        aligned_functional_draws(&xs, &ys, ValueFunctional::Product, &token()).expect("rows");
    let mapped = aligned_map_draws(&[&xs, &ys], &|r| r[0] * r[1], &token()).expect("mapped rows");
    assert_eq!(rows, mapped);
    let spread = draw_spread(&rows, None).expect("spread");
    // Rows are 2 twice and 0 eight times: mean 0.4, population variance
    // 0.2 * 4 - 0.16 = 0.64, sample variance 0.64 * 10 / 9.
    assert!((spread.mean - 0.4).abs() < 1e-15);
    assert!((spread.variance - 0.64 * 10.0 / 9.0).abs() < 1e-14);
    // The same columns paired apart by a rotation are another joint law
    // (E[XY] = 0.6 here): the spread is a property of the aligned rows,
    // not of the marginals.
    let shifted: Vec<f64> = ys.iter().cycle().skip(3).take(ys.len()).copied().collect();
    let unpaired =
        aligned_functional_draws(&xs, &shifted, ValueFunctional::Product, &token()).expect("rows");
    assert!((draw_spread(&unpaired, None).expect("spread").mean - spread.mean).abs() > 1e-12);
}

#[test]
fn interval_candidates_are_the_closed_form_and_carry_no_coverage() {
    let normal = normal_interval_candidate(10.0, 4.0, 0.95).expect("normal");
    assert!((normal.lower - (10.0 - 2.0 * 1.959_963_984_540_054)).abs() < 1e-9);
    assert!((normal.upper - (10.0 + 2.0 * 1.959_963_984_540_054)).abs() < 1e-9);
    assert_eq!(normal.calibration, "not_measured");
    assert_eq!(normal.construction, NORMAL_APPROXIMATION_CONSTRUCTION);
    // Draws 0..=100: type 7 ranks p (n - 1), type 6 ranks p (n + 1).
    let draws: Vec<f64> = (0..=100).map(f64::from).collect();
    let interpolated =
        draw_interval_candidate(&draws, 0.9, QuantileRule::Interpolated).expect("type 7");
    assert!((interpolated.lower - 5.0).abs() < 1e-12 && (interpolated.upper - 95.0).abs() < 1e-12);
    let exchangeable =
        draw_interval_candidate(&draws, 0.9, QuantileRule::ExchangeableRank).expect("type 6");
    // one-based rank 0.05 * 102 = 5.1 -> value 4.1; 0.95 * 102 = 96.9 -> 95.9.
    assert!((exchangeable.lower - 4.1).abs() < 1e-12);
    assert!((exchangeable.upper - 95.9).abs() < 1e-12);
    assert_ne!(interpolated.construction, exchangeable.construction);
    for level in [0.0, 1.0, f64::NAN, -0.1] {
        assert!(matches!(
            normal_interval_candidate(0.0, 1.0, level),
            Err(JointLawError::InvalidLevel(_))
        ));
    }
    assert!(matches!(
        normal_interval_candidate(0.0, -1.0, 0.9),
        Err(JointLawError::InvalidVariance(_))
    ));
}

#[test]
fn mixture_law_quantiles_and_exceedance_use_the_weights_only() {
    // 0.5 [0, 1] + 0.5 [1, 2]: atoms 0 (0.25), 1 (0.5), 2 (0.25).
    let a = [0.0, 1.0];
    let b = [1.0, 2.0];
    let law = mixture_marginal(&[0.5, 0.5], &[&a, &b]).expect("law");
    assert_eq!(law.atoms(), &[(0.0, 0.25), (1.0, 0.5), (2.0, 0.25)]);
    assert!((law.quantile(0.25).expect("q") - 0.0).abs() < 1e-12);
    assert!((law.quantile(0.26).expect("q") - 1.0).abs() < 1e-12);
    assert!((law.quantile(0.75).expect("q") - 1.0).abs() < 1e-12);
    assert!((law.quantile(0.76).expect("q") - 2.0).abs() < 1e-12);
    assert!((law.quantile(1.0).expect("q") - 2.0).abs() < 1e-12);
    assert!(law.quantile(0.0).is_err());
    let candidate = marginal_interval_candidate(&law, 0.5).expect("interval");
    // Level 0.5: the 0.25 and 0.75 quantiles, 0 and 1.
    assert_eq!((candidate.lower, candidate.upper), (0.0, 1.0));
    // Unequal component sizes keep the declared weights, not the counts.
    let heavy = [10.0; 6];
    let skewed = mixture_marginal(&[0.25, 0.75], &[&a[..], &heavy[..]]).expect("law");
    assert!((skewed.mean() - (0.25 * 0.5 + 0.75 * 10.0)).abs() < 1e-14);
    // Exceedance: components exceed 0.5 with probability 0.5 and 0.75.
    let c1 = [0.0, 0.0, 1.0, 1.0];
    let c2 = [1.0, 1.0, 1.0, 0.0];
    let (p, se) = mixture_exceedance(&[0.25, 0.75], &[&c1, &c2], 0.5).expect("p");
    assert!((p - 0.6875).abs() < 1e-15);
    let var: f64 = 0.0625 * 0.25 / 4.0 + 0.5625 * 0.1875 / 4.0;
    assert!((se - var.sqrt()).abs() < 1e-15);
    assert!(matches!(
        mixture_exceedance(&[0.5, 0.4], &[&c1, &c2], 0.5),
        Err(JointLawError::InvalidWeights(_))
    ));
}

#[test]
fn exceedance_decisions_report_the_monte_carlo_band() {
    // 90 of 100 rows above the threshold.
    let mut draws = vec![1.0; 90];
    draws.extend(vec![-1.0; 10]);
    let d = composed_exceedance_decision(&draws, 0.0, 0.5, 0.95, None).expect("decision");
    assert!((d.probability - 0.9).abs() < 1e-15);
    // Adjusted proportion (90 + 1.9207) / (100 + 3.8415) -> se 0.031284.
    assert!((d.standard_error - 0.031_284).abs() < 1e-5, "{d:?}");
    assert!((d.band - 1.959_964 * d.standard_error).abs() < 1e-8);
    assert_eq!(d.verdict, DecisionVerdict::Act);
    assert_eq!(d.calibration, "not_measured");
    for (cutoff, expected) in [
        (0.5, DecisionVerdict::Act),
        (0.9, DecisionVerdict::Unresolved),
        (0.99, DecisionVerdict::Withhold),
    ] {
        let at = composed_exceedance_decision(&draws, 0.0, cutoff, 0.95, None).expect("d");
        assert_eq!(at.verdict, expected, "cutoff {cutoff}");
    }
    // Boundary: every row above the threshold still has a nonzero band, so
    // a cutoff just under one is unresolved rather than act.
    let all = vec![1.0; 100];
    let at_one = composed_exceedance_decision(&all, 0.0, 0.99, 0.95, None).expect("d");
    assert!((at_one.probability - 1.0).abs() < 1e-12);
    assert!(at_one.standard_error > 0.013 && at_one.standard_error < 0.0133);
    assert_eq!(at_one.verdict, DecisionVerdict::Unresolved);
    assert_eq!(
        composed_exceedance_decision(&all, 0.0, 0.95, 0.95, None).expect("d").verdict,
        DecisionVerdict::Act
    );
    // Fewer effective draws widen the band and can unresolve a decision.
    let thin = composed_exceedance_decision(&draws, 0.0, 0.5, 0.95, Some(10.0)).expect("d");
    assert!(thin.band > d.band);
    assert!(matches!(
        composed_exceedance_decision(&draws, 0.0, 1.0, 0.95, None),
        Err(JointLawError::InvalidLevel(_))
    ));
    assert!(matches!(
        composed_exceedance_decision(&[], 0.0, 0.5, 0.95, None),
        Err(JointLawError::Shape(_))
    ));
}

#[test]
fn cancellation_is_observed_in_the_coupling_walk() {
    let draws: Vec<f64> = (0..20_000).map(f64::from).collect();
    let m = Marginal::from_draws(&draws).expect("m");
    let c = CancellationToken::new();
    c.cancel();
    assert_eq!(product_moment_envelope(&m, &m, &c), Err(JointLawError::Cancelled));
}

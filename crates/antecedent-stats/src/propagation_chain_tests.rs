//! Known-truth checks of the chain arithmetic, each against a value derived by
//! hand or from a closed form.

use super::*;

fn token() -> CancellationToken {
    CancellationToken::new()
}

/// Mid-quantile grid of `N(mean, sd^2)`: `rows` equally likely draws.
fn normal_grid(rows: usize, mean: f64, sd: f64) -> Vec<f64> {
    (0..rows).map(|i| mean + sd * normal_ppf((i as f64 + 0.5) / rows as f64)).collect()
}

/// Rows and stride of a Fibonacci lattice: pairing a grid with a copy of itself
/// visited at this stride gives draws of two independent variables that a
/// smooth function integrates almost exactly.
const FIBONACCI_ROWS: usize = 1597;
const FIBONACCI_STRIDE: usize = 987;

/// The same grid visited in a fixed scrambled order. At the Fibonacci stride the
/// scrambled column and the original are decorrelated draws of independent
/// variables.
fn scrambled(grid: &[f64], stride: usize) -> Vec<f64> {
    (0..grid.len()).map(|i| grid[(i * stride) % grid.len()]).collect()
}

fn near(got: f64, want: f64, tolerance: f64) {
    assert!((got - want).abs() <= tolerance, "{got} vs {want} (tolerance {tolerance})");
}

fn unary(input: usize, function: UnaryFn) -> StageSpec {
    StageSpec { inputs: vec![input], op: StageOp::Unary(function) }
}

#[test]
fn affine_chain_has_the_known_mean_and_variance() {
    // x = 1,2,3,4: mean 2.5, sample variance 5/3.
    let sources = vec![vec![1.0, 2.0, 3.0, 4.0]];
    let stages = vec![
        StageSpec {
            inputs: vec![0],
            op: StageOp::Affine { scale: Coefficient::Fixed(2.0), shift: Coefficient::Fixed(1.0) },
        },
        StageSpec {
            inputs: vec![1],
            op: StageOp::Affine { scale: Coefficient::Fixed(-3.0), shift: Coefficient::Fixed(0.0) },
        },
        unary(2, UnaryFn::Square),
    ];
    let evaluated = evaluate_columns(&sources, &stages, &token()).expect("evaluates");
    let second = evaluated.columns[1].as_ref().expect("column");
    let summary = summarize(second, None).expect("summary");
    // 2x+1 has mean 6, variance 4 * 5/3; times -3 gives mean -18, variance 9 * 20/3.
    near(summary.mean, -18.0, 1e-12);
    near(summary.variance, 60.0, 1e-12);
    // The square is the nonlinear stage: E[(−3(2x+1))^2] = Var + mean^2 (population form
    // for the equal-weight rows): rows are -9,-15,-21,-27 -> squares 81,225,441,729.
    let square = evaluated.columns[2].as_ref().expect("column");
    near(
        summarize(square, None).expect("summary").mean,
        (81.0 + 225.0 + 441.0 + 729.0) / 4.0,
        1e-9,
    );
}

#[test]
fn nonlinear_stage_is_not_the_stage_of_the_mean() {
    let sources = vec![normal_grid(4000, 1.0, 0.8)];
    let stages = vec![
        unary(0, UnaryFn::ExponentialUtility { risk_aversion: 0.5 }),
        unary(0, UnaryFn::Exp),
        unary(0, UnaryFn::Square),
    ];
    let evaluated = evaluate_columns(&sources, &stages, &token()).expect("evaluates");
    let utility = summarize(evaluated.columns[0].as_ref().expect("column"), None).expect("summary");
    // E[1 - exp(-r X)] = 1 - exp(-r mu + r^2 s^2 / 2) for a normal X.
    let truth = 1.0 - (-0.5_f64 * 1.0 + 0.25 * 0.64 / 2.0).exp();
    near(utility.mean, truth, 2e-3);
    // The utility of the mean is a different number: g(E[X]) != E[g(X)].
    let at_mean = 1.0 - (-0.5_f64 * 1.0).exp();
    assert!((at_mean - truth).abs() > 0.03);
    // A lognormal mean exp(mu + s^2/2) and the second moment mu^2 + s^2.
    let lognormal =
        summarize(evaluated.columns[1].as_ref().expect("column"), None).expect("summary");
    near(lognormal.mean, (1.0_f64 + 0.32).exp(), 2e-2);
    let square = summarize(evaluated.columns[2].as_ref().expect("column"), None).expect("summary");
    near(square.mean, 1.0 + 0.64, 2e-2);
}

#[test]
fn uncertain_coefficient_columns_enter_the_joint_law() {
    // y = slope * x + intercept with independent uncertain slope and x. For independent
    // normals E[slope x] = E[slope] E[x] but Var[slope x] = (s_a^2 + m_a^2)(s_x^2 + m_x^2) - (m_a m_x)^2.
    let rows = FIBONACCI_ROWS;
    let x = normal_grid(rows, 2.0, 0.5);
    let slope = scrambled(&normal_grid(rows, 3.0, 0.4), FIBONACCI_STRIDE);
    let sources = vec![x, slope];
    let stages = vec![StageSpec {
        inputs: vec![0, 1],
        op: StageOp::Affine { scale: Coefficient::Input(1), shift: Coefficient::Fixed(1.0) },
    }];
    let evaluated = evaluate_columns(&sources, &stages, &token()).expect("evaluates");
    let summary = summarize(evaluated.columns[0].as_ref().expect("column"), None).expect("summary");
    near(summary.mean, 7.0, 5e-2);
    let variance = (0.16 + 9.0) * (0.25 + 4.0) - 36.0;
    near(summary.variance, variance, 0.2);
}

#[test]
fn a_dependent_pool_counts_shared_evidence_once() {
    // T, e1, e2 over the 2^3 sign design: orthogonal columns with sample variance 8/7.
    let mut t = Vec::new();
    let mut e1 = Vec::new();
    let mut e2 = Vec::new();
    for a in [-1.0, 1.0] {
        for b in [-1.0, 1.0] {
            for c in [-1.0, 1.0] {
                t.push(a);
                e1.push(b);
                e2.push(c);
            }
        }
    }
    let a: Vec<f64> = t.iter().zip(&e1).map(|(x, y)| 10.0 + x + y).collect();
    let b: Vec<f64> = t.iter().zip(&e2).map(|(x, y)| 10.0 + x + y).collect();
    let sources = vec![a, b];
    let pool = |dependent: bool| {
        let stages = vec![StageSpec { inputs: vec![0, 1], op: StageOp::Pool { dependent } }];
        evaluate_columns(&sources, &stages, &token()).expect("evaluates")
    };
    let dependent = pool(true);
    let outcome = dependent.pools[0].as_ref().expect("pool");
    // Var A = Var B = 16/7, Cov = 8/7; equal GLS weights; pooled variance (v + c)/2 = 12/7.
    near(outcome.weights[0], 0.5, 1e-12);
    near(outcome.weights[1], 0.5, 1e-12);
    near(outcome.joint_variance, 12.0 / 7.0, 1e-12);
    // The independent claim is 1/(1/v + 1/v) = 8/7: it counts the shared T twice.
    near(outcome.independent_variance, 8.0 / 7.0, 1e-12);
    assert!(outcome.independent_variance < outcome.joint_variance);
}

#[test]
fn an_independent_pool_of_independent_inputs_uses_inverse_variance_weights() {
    let rows = FIBONACCI_ROWS;
    let a = normal_grid(rows, 5.0, 1.0);
    let b = scrambled(&normal_grid(rows, 5.0, 2.0), FIBONACCI_STRIDE);
    let sources = vec![a, b];
    let stages = vec![StageSpec { inputs: vec![0, 1], op: StageOp::Pool { dependent: false } }];
    let evaluated = evaluate_columns(&sources, &stages, &token()).expect("evaluates");
    let outcome = evaluated.pools[0].as_ref().expect("pool");
    near(outcome.weights[0], 0.8, 1e-2);
    near(outcome.weights[1], 0.2, 1e-2);
    // 1/(1/1 + 1/4) = 0.8 under independence; the joint law agrees up to grid error.
    near(outcome.independent_variance, 0.8, 5e-2);
    near(outcome.joint_variance, 0.8, 8e-2);
}

#[test]
fn a_dependent_pool_of_duplicated_evidence_is_singular() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![2.0, 4.0, 6.0, 8.0];
    let stages = vec![StageSpec { inputs: vec![0, 1], op: StageOp::Pool { dependent: true } }];
    let refusal = evaluate_columns(&[a, b], &stages, &token()).expect_err("refused");
    assert_eq!(refusal, ChainError::SingularPool);
}

#[test]
fn weights_reweight_the_summary_and_equal_weights_reproduce_it() {
    let column = vec![0.0, 1.0, 2.0, 3.0];
    let plain = summarize(&column, None).expect("summary");
    let equal = summarize(&column, Some(&[0.25; 4])).expect("summary");
    assert_eq!(plain, equal);
    near(plain.variance, 5.0 / 3.0, 1e-12);
    // All the mass on row 3 moves the mean there and leaves one effective row.
    let heavy = weights_from_log(&[0.0, 0.0, 0.0, 800.0]).expect("weights");
    let shifted = summarize(&column, Some(&heavy)).expect("summary");
    near(shifted.mean, 3.0, 1e-9);
    assert!(shifted.effective_size < 1.0 + 1e-6);
    assert!(weights_from_log(&[0.0, f64::NAN]).is_err());
}

#[test]
fn a_decision_verdict_needs_the_gap_to_clear_the_band() {
    let rows = 1000;
    let utility = normal_grid(rows, 0.5, 1.0);
    let summary = summarize(&utility, None).expect("summary");
    assert_eq!(decide(&summary, 0.0).verdict, DecisionVerdict::Act);
    assert_eq!(decide(&summary, 2.0).verdict, DecisionVerdict::Withhold);
    assert_eq!(decide(&summary, 0.5).verdict, DecisionVerdict::Unresolved);
}

#[test]
fn preposterior_value_matches_the_closed_form() {
    // theta ~ N(1/2, 1), a study of n = 4 observations with variance 12, a linear
    // utility with reference 0. The study's posterior mean has variance
    // tau^2 - tau'^2 with tau'^2 = 1/(1 + n/sigma^2) = 3/4, so its sd is 1/2 and
    // EVSI = sd * (phi(d) - d * Phi(-d)) with d = (1/2)/(1/2) = 1.
    let rows = FIBONACCI_ROWS;
    let theta = normal_grid(rows, 0.5, 1.0);
    let noise = scrambled(&normal_grid(rows, 0.0, 1.0), FIBONACCI_STRIDE);
    let study =
        Preposterior { noise_sd: 12.0_f64.sqrt(), sample_size: 4.0, reference_utility: 0.0, noise };
    let outcome = preposterior(&theta, &theta, None, &study, &token()).expect("update");
    let d = 1.0;
    let evsi = 0.5 * (antecedent_kernels::norm_pdf(d) - d * antecedent_kernels::norm_cdf(-d));
    near(outcome.value_of_information, evsi, 2e-2);
    // Without the study the decision is to act (mean 1/2): the update can only help.
    near(outcome.current_expected_utility, 0.5, 1e-3);
    near(outcome.mean_updated_expected_utility, outcome.current_expected_utility, 2e-2);
    assert!(outcome.probability_decision_changes > 0.05);
    assert!(outcome.mean_effective_size > MIN_UPDATE_EFFECTIVE_SIZE);
}

#[test]
fn an_uninformative_study_changes_nothing() {
    let rows = 600;
    let theta = normal_grid(rows, 0.5, 1.0);
    let noise = scrambled(&normal_grid(rows, 0.0, 1.0), 307);
    let study = Preposterior { noise_sd: 1.0e4, sample_size: 1.0, reference_utility: 0.0, noise };
    let outcome = preposterior(&theta, &theta, None, &study, &token()).expect("update");
    near(outcome.value_of_information, 0.0, 1e-3);
    near(outcome.probability_decision_changes, 0.0, 1e-9);
}

#[test]
fn a_study_that_leaves_too_few_updated_draws_is_refused() {
    let rows = 200;
    let theta = normal_grid(rows, 0.0, 1.0);
    let noise = scrambled(&normal_grid(rows, 0.0, 1.0), 31);
    let study = Preposterior { noise_sd: 1.0e-4, sample_size: 1.0, reference_utility: 0.0, noise };
    let refusal = preposterior(&theta, &theta, None, &study, &token()).expect_err("refused");
    assert!(matches!(refusal, ChainError::LowEffectiveSize { .. }));
}

#[test]
fn enclosures_are_the_image_of_the_box_under_every_coupling() {
    let sources = [
        Enclosure::new(1.0, 2.0).expect("enclosure"),
        Enclosure::new(-1.0, 3.0).expect("enclosure"),
    ];
    let stages = vec![
        unary(0, UnaryFn::Exp),
        unary(1, UnaryFn::Square),
        StageSpec { inputs: vec![0, 1], op: StageOp::Binary(Combine::Product) },
    ];
    let out = evaluate_enclosures(&sources, &stages).expect("enclosures");
    near(out[0].lower, 1.0_f64.exp(), 1e-12);
    near(out[0].upper, 2.0_f64.exp(), 1e-12);
    // The square of [-1, 3] reaches its minimum at zero inside the interval.
    near(out[1].lower, 0.0, 1e-12);
    near(out[1].upper, 9.0, 1e-12);
    // [1, 2] * [-1, 3] is [-2, 6].
    near(out[2].lower, -2.0, 1e-12);
    near(out[2].upper, 6.0, 1e-12);
}

#[test]
fn an_enclosure_refuses_a_divisor_reaching_zero_and_a_pool() {
    let sources = [
        Enclosure::new(1.0, 2.0).expect("enclosure"),
        Enclosure::new(-1.0, 1.0).expect("enclosure"),
    ];
    let ratio = vec![StageSpec { inputs: vec![0, 1], op: StageOp::Binary(Combine::Ratio) }];
    assert!(matches!(
        evaluate_enclosures(&sources, &ratio),
        Err(ChainError::EnclosureUnbounded(_))
    ));
    let pool = vec![StageSpec { inputs: vec![0, 1], op: StageOp::Pool { dependent: false } }];
    assert!(matches!(evaluate_enclosures(&sources, &pool), Err(ChainError::Operator(_))));
}

#[test]
fn a_stage_that_leaves_its_domain_is_refused() {
    let stages = vec![unary(0, UnaryFn::Log)];
    let refusal =
        evaluate_columns(&[vec![1.0, -1.0, 2.0]], &stages, &token()).expect_err("refused");
    assert!(matches!(refusal, ChainError::Domain(_)));
}

#[test]
fn malformed_chains_are_refused_before_any_number() {
    // A stage may not read itself or a later node.
    let forward = vec![StageSpec { inputs: vec![1], op: StageOp::Unary(UnaryFn::Exp) }];
    assert!(validate_spec(1, &forward).is_err());
    // A coefficient is an input after the first.
    let coefficient = vec![StageSpec {
        inputs: vec![0],
        op: StageOp::Affine { scale: Coefficient::Input(0), shift: Coefficient::Fixed(0.0) },
    }];
    assert!(validate_spec(1, &coefficient).is_err());
    // An update is terminal.
    let study =
        Preposterior { noise_sd: 1.0, sample_size: 1.0, reference_utility: 0.0, noise: vec![] };
    let after_update = vec![
        StageSpec { inputs: vec![0, 0], op: StageOp::Preposterior(study) },
        unary(1, UnaryFn::Exp),
    ];
    assert!(validate_spec(1, &after_update).is_err());
}

#[test]
fn canonical_tags_are_distinct_per_operator() {
    let digest = |op: &StageOp| {
        let mut writer = CanonicalWriter::new("test");
        op.encode(&mut writer);
        writer.finish()
    };
    let ops = [
        StageOp::Pool { dependent: false },
        StageOp::Pool { dependent: true },
        StageOp::Binary(Combine::Sum),
        StageOp::Binary(Combine::Product),
        StageOp::Unary(UnaryFn::Exp),
        StageOp::Unary(UnaryFn::Log),
    ];
    for (i, a) in ops.iter().enumerate() {
        for b in ops.iter().skip(i + 1) {
            assert_ne!(digest(a), digest(b));
        }
    }
}

#[test]
fn an_independent_product_table_holds_every_combination_once() {
    let table =
        independent_product_table(&[vec![1.0, 2.0], vec![10.0, 20.0, 30.0]], 100).expect("table");
    assert_eq!(table[0], vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    assert_eq!(table[1], vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
    // The columns are uncorrelated by construction, whatever the draws.
    let a = summarize(&table[0], None).expect("summary");
    let b = summarize(&table[1], None).expect("summary");
    let cross: f64 =
        table[0].iter().zip(&table[1]).map(|(x, y)| (x - a.mean) * (y - b.mean)).sum::<f64>();
    near(cross, 0.0, 1e-12);
    // A product larger than the budget is refused rather than truncated.
    assert!(matches!(
        independent_product_table(&[vec![1.0; 10], vec![1.0; 10]], 50),
        Err(ChainError::Budget(_))
    ));
}

#[test]
fn a_product_table_feeds_a_nonlinear_stage_the_joint_law_of_independent_draws() {
    // X in {0, 2} and Y in {0, 2}, each with probability 1/2, independent: E[XY] = 1.
    let table = independent_product_table(&[vec![0.0, 2.0], vec![0.0, 2.0]], 100).expect("table");
    let stages = vec![StageSpec { inputs: vec![0, 1], op: StageOp::Binary(Combine::Product) }];
    let evaluated = evaluate_columns(&table, &stages, &token()).expect("evaluates");
    let product = summarize(evaluated.columns[0].as_ref().expect("column"), None).expect("summary");
    near(product.mean, 1.0, 1e-12);
}

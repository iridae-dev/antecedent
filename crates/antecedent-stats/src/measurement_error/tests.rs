use super::*;
use antecedent_core::CausalRng;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// Standard normal by Box-Muller from the library `SplitMix` stream.
fn normal(rng: &mut CausalRng) -> f64 {
    let u1 = 1.0 - rng.next_f64();
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Full factorial over three balanced binary factors X in {-a,a}, U in
/// {-b,b}, E in {-c,c} (n = 8): all pairwise sample covariances are
/// exactly zero, so every moment has a closed form.
fn factorial(a: f64, b: f64, c: f64, beta: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x = Vec::new();
    let mut u = Vec::new();
    let mut e = Vec::new();
    for sx in [-1.0, 1.0] {
        for su in [-1.0, 1.0] {
            for se in [-1.0, 1.0] {
                x.push(sx * a);
                u.push(su * b);
                e.push(se * c);
            }
        }
    }
    let w: Vec<f64> = x.iter().zip(&u).map(|(x, u)| x + u).collect();
    let y: Vec<f64> = x.iter().zip(&e).map(|(x, e)| 2.0 + beta * x + e).collect();
    (y, w, x)
}

#[test]
fn closed_form_attenuation_is_removed_exactly() {
    // lambda = a^2 / (a^2 + b^2) = 1 / (1 + 0.25) = 0.8 with a = 1, b = 0.5.
    let (y, w, _) = factorial(1.0, 0.5, 0.7, 3.0);
    let data = LinearErrorData {
        y: &y,
        w: &w,
        covariates: &[],
        covariate_count: 0,
        replicate_counts: None,
    };
    let error = ErrorVariance::known(0.25, 0.0).unwrap();
    let moments = LinearMoments::accumulate(&data, &error, 0.01).unwrap();
    let fit = moments.solve().unwrap();
    assert!(close(fit.naive_coefficients[1], 3.0 * 0.8, 1e-12), "{:?}", fit.naive_coefficients);
    assert!(close(fit.coefficients[1], 3.0, 1e-12), "{:?}", fit.coefficients);
    assert!(close(fit.coefficients[0], 2.0, 1e-12));
    assert!(close(fit.reliability, 0.8, 1e-12));
}

#[test]
fn corrected_normal_equations_equal_regression_calibration() {
    // Independent route: substitute E[X | W] = mean(W) + lambda (W - mean(W)),
    // lambda = (s_w^2 - sigma_u^2) / s_w^2, then run simple least squares.
    let mut rng = CausalRng::from_seed(11);
    let n = 400;
    let sigma_u2: f64 = 0.6;
    let x: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
    let w: Vec<f64> = x.iter().map(|v| v + sigma_u2.sqrt() * normal(&mut rng)).collect();
    let y: Vec<f64> = x.iter().map(|v| 1.0 + 2.0 * v + normal(&mut rng)).collect();
    let nf = f64::from(n);
    let w_bar = w.iter().sum::<f64>() / nf;
    let s_w2 = w.iter().map(|v| (v - w_bar) * (v - w_bar)).sum::<f64>() / nf;
    let lambda = (s_w2 - sigma_u2) / s_w2;
    let x_hat: Vec<f64> = w.iter().map(|v| w_bar + lambda * (v - w_bar)).collect();
    let xh_bar = x_hat.iter().sum::<f64>() / nf;
    let y_bar = y.iter().sum::<f64>() / nf;
    let sxy: f64 = x_hat.iter().zip(&y).map(|(a, b)| (a - xh_bar) * (b - y_bar)).sum();
    let sxx: f64 = x_hat.iter().map(|a| (a - xh_bar) * (a - xh_bar)).sum();
    let rc_slope = sxy / sxx;
    let data = LinearErrorData {
        y: &y,
        w: &w,
        covariates: &[],
        covariate_count: 0,
        replicate_counts: None,
    };
    let moments =
        LinearMoments::accumulate(&data, &ErrorVariance::known(sigma_u2, 0.0).unwrap(), 0.01)
            .unwrap();
    let fit = moments.solve().unwrap();
    assert!(close(fit.coefficients[1], rc_slope, 1e-10), "{} vs {rc_slope}", fit.coefficients[1]);
}

#[test]
fn variance_includes_error_variance_uncertainty() {
    let (y, w, _) = factorial(1.0, 0.5, 0.7, 3.0);
    let data = LinearErrorData {
        y: &y,
        w: &w,
        covariates: &[],
        covariate_count: 0,
        replicate_counts: None,
    };
    let exact = LinearMoments::accumulate(&data, &ErrorVariance::known(0.25, 0.0).unwrap(), 0.01)
        .unwrap()
        .solve()
        .unwrap();
    let uncertain =
        LinearMoments::accumulate(&data, &ErrorVariance::known(0.25, 0.05).unwrap(), 0.01)
            .unwrap()
            .solve()
            .unwrap();
    assert_eq!(exact.coefficients, uncertain.coefficients);
    assert!(close(exact.covariance[3], exact.covariance_given_error_variance[3], 1e-15));
    // d beta / d sigma^2 for the scalar slope: beta / (s_w^2 - sigma^2) = 3 / 1.0.
    let added = uncertain.covariance[3] - uncertain.covariance_given_error_variance[3];
    assert!(close(added, (3.0_f64 / 1.0).powi(2) * 0.05 * 0.05, 1e-10), "{added}");
}

#[test]
fn reliability_at_or_below_floor_refuses() {
    let (y, w, _) = factorial(1.0, 0.5, 0.7, 3.0);
    let data = LinearErrorData {
        y: &y,
        w: &w,
        covariates: &[],
        covariate_count: 0,
        replicate_counts: None,
    };
    // Var(W) = 1.25: an error variance of 1.25 leaves no latent variance.
    let at_total = ErrorVariance::known(1.25, 0.0).unwrap();
    assert!(matches!(
        LinearMoments::accumulate(&data, &at_total, 0.0),
        Err(MeasurementErrorError::ReliabilityNotPositive { .. })
    ));
    // 0.8 reliability passes a floor of 0.5 and fails a floor of 0.9.
    let ok = ErrorVariance::known(0.25, 0.0).unwrap();
    assert!(LinearMoments::accumulate(&data, &ok, 0.5).is_ok());
    assert!(matches!(
        LinearMoments::accumulate(&data, &ok, 0.9),
        Err(MeasurementErrorError::ReliabilityNotPositive { .. })
    ));
}

#[test]
fn replicate_components_match_the_balanced_anova_closed_form() {
    // Two units, k = 2: values (1, 3) and (5, 9).
    // unit means 2, 7; grand 4.5; SSW = 2 + 8 = 10 on 2 df -> MSW = 5.
    // SSB = 2*(2.5^2) + 2*(2.5^2) = 25 on 1 df -> MSB = 25; n0 = 2.
    // between = (25 - 5) / 2 = 10; ICC = 10 / 15.
    let (components, influence) =
        replicate_variance_components(&[1.0, 3.0, 5.0, 9.0], &[2, 2]).unwrap();
    assert!(close(components.within_variance, 5.0, 1e-12));
    assert!(close(components.between_variance, 10.0, 1e-12));
    assert!(close(components.intraclass_correlation, 10.0 / 15.0, 1e-12));
    assert!(close(components.effective_replicates, 2.0, 1e-12));
    // Influence: (SSW_j - 1 * 5) / mean(k - 1) = (2 - 5, 8 - 5) / 1.
    assert!(close(influence[0], -3.0, 1e-12) && close(influence[1], 3.0, 1e-12));
    assert!(replicate_variance_components(&[1.0, 2.0], &[1, 1]).is_err());
}

#[test]
fn simulated_replicates_recover_truth_and_the_reported_se_matches_the_spread() {
    // DGP: X ~ N(0,1), Z = 0.5 X + N(0,1), Y = 1 + 1.5 X + 0.8 Z + N(0,1),
    // W_ij = X_i + U_ij with var(U) = 0.7, k = 3 replicates.
    // lambda (single) = 1 / 1.7; naive slope is attenuated well below 1.5.
    let reps = 300_u32;
    let n = 1000;
    let k = 3_u32;
    let mut estimates = Vec::new();
    let mut se_sum = 0.0;
    let mut naive_sum = 0.0;
    for rep in 0..reps {
        let mut rng = CausalRng::from_seed(1_000 + u64::from(rep));
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        let mut wbar = Vec::with_capacity(n);
        let mut flat = Vec::with_capacity(n * k as usize);
        for _ in 0..n {
            let x = normal(&mut rng);
            let zi = 0.5 * x + normal(&mut rng);
            y.push(1.0 + 1.5 * x + 0.8 * zi + normal(&mut rng));
            z.push(zi);
            let mut sum = 0.0;
            for _ in 0..k {
                let w = x + 0.7_f64.sqrt() * normal(&mut rng);
                flat.push(w);
                sum += w;
            }
            wbar.push(sum / f64::from(k));
        }
        let sizes = vec![k; n];
        let (error, _) = ErrorVariance::from_main_sample_replicates(&flat, &sizes).unwrap();
        let data = LinearErrorData {
            y: &y,
            w: &wbar,
            covariates: &z,
            covariate_count: 1,
            replicate_counts: Some(&sizes),
        };
        let fit = LinearMoments::accumulate(&data, &error, 0.05).unwrap().solve().unwrap();
        estimates.push(fit.coefficients[1]);
        se_sum += fit.standard_error(1);
        naive_sum += fit.naive_coefficients[1];
    }
    let mean = estimates.iter().sum::<f64>() / f64::from(reps);
    let sd = (estimates.iter().map(|e| (e - mean) * (e - mean)).sum::<f64>()
        / (f64::from(reps) - 1.0))
        .sqrt();
    // Predeclared: |mean - 1.5| within 4 spreads of the mean of `reps` fits.
    assert!(close(mean, 1.5, 4.0 * sd / f64::from(reps).sqrt() + 0.01), "mean {mean} sd {sd}");
    // The naive coefficient is attenuated: strictly below the truth by a wide margin.
    assert!(naive_sum / f64::from(reps) < 1.3, "naive {}", naive_sum / f64::from(reps));
    // Mean reported SE tracks the empirical spread (predeclared band 0.8..1.25).
    let ratio = (se_sum / f64::from(reps)) / sd;
    assert!((0.8..=1.25).contains(&ratio), "se/sd ratio {ratio}");
}

#[test]
fn linear_functional_variance_is_the_quadratic_form_of_the_joint_covariance() {
    let (y, w, _) = factorial(1.0, 0.5, 0.7, 3.0);
    let data = LinearErrorData {
        y: &y,
        w: &w,
        covariates: &[],
        covariate_count: 0,
        replicate_counts: None,
    };
    let fit = LinearMoments::accumulate(&data, &ErrorVariance::known(0.25, 0.1).unwrap(), 0.01)
        .unwrap()
        .solve()
        .unwrap();
    let (value, variance) = fit.linear_functional(&[0.0, 2.0]).unwrap();
    assert!(close(value, 6.0, 1e-12));
    assert!(close(variance, 4.0 * fit.covariance[3], 1e-12));
    let (v2, var2) = fit.linear_functional(&[1.0, 1.0]).unwrap();
    let expect = fit.covariance[0] + 2.0 * fit.covariance[1] + fit.covariance[3];
    assert!(close(v2, 5.0, 1e-12) && close(var2, expect, 1e-12));
    assert!(fit.linear_functional(&[1.0]).is_err());
}

#[test]
fn rogan_gladen_inverts_the_misclassification_algebra() {
    // pi = 0.3, Se = 0.9, Sp = 0.8: p* = 0.9*0.3 + 0.2*0.7 = 0.41.
    let rates = BinaryErrorRates::new(0.9, 0.8, 0.0001, 0.0004).unwrap();
    let result = corrected_prevalence(410, 1000, &rates, 0.1).unwrap();
    assert!(close(result.observed, 0.41, 1e-15));
    assert!(close(result.corrected, 0.3, 1e-12));
    assert!(close(result.youden, 0.7, 1e-12));
    // Var = [p*(1-p*)/n + (1-pi)^2 Var(Sp) + pi^2 Var(Se)] / J^2.
    let expect = (0.41 * 0.59 / 1000.0 + 0.49 * 0.0004 + 0.09 * 0.0001) / 0.49;
    assert!(close(result.variance, expect, 1e-12), "{} vs {expect}", result.variance);
    assert!(close(result.variance_sampling + result.variance_rates, result.variance, 1e-15));
    assert!(!result.outside_unit_interval);
}

#[test]
fn observed_class_counts_that_overflow_u64_refuse() {
    let confusion =
        ConfusionMatrix::new(2, vec![0.9, 0.2, 0.1, 0.8], Some(vec![200, 300])).unwrap();
    let classes = [
        ObservedClassSummary { count: 1 << 63, mean: 1.0, variance: 0.5 },
        ObservedClassSummary { count: (1 << 63) + 2, mean: 0.0, variance: 0.5 },
    ];
    assert!(matches!(
        corrected_categories(&classes, &confusion, 20.0),
        Err(MeasurementErrorError::InvalidParameter(_))
    ));
}

#[test]
fn youden_at_or_below_the_declared_minimum_refuses() {
    // Se + Sp - 1 = 0 (a coin flip) and a negative index both refuse.
    for (se, sp) in [(0.5, 0.5), (0.3, 0.6)] {
        let rates = BinaryErrorRates::new(se, sp, 0.0, 0.0).unwrap();
        assert!(matches!(
            corrected_prevalence(10, 100, &rates, 0.05),
            Err(MeasurementErrorError::YoudenTooSmall { .. })
        ));
    }
    // 0.12 passes a minimum of 0.1 and fails a minimum of 0.2.
    let weak = BinaryErrorRates::new(0.56, 0.56, 0.0, 0.0).unwrap();
    assert!(corrected_prevalence(10, 100, &weak, 0.1).is_ok());
    assert!(corrected_prevalence(10, 100, &weak, 0.2).is_err());
}

#[test]
fn nondifferential_risk_difference_has_closed_form_and_shared_rate_variance() {
    // pi1 = 0.4, pi0 = 0.2, Se = 0.9, Sp = 0.8:
    // p1* = 0.36 + 0.12 = 0.48, p0* = 0.18 + 0.16 = 0.34.
    let rates = BinaryErrorRates::new(0.9, 0.8, 0.0004, 0.0009).unwrap();
    let table = TwoByTwoCounts {
        exposed_events: 480,
        exposed_total: 1000,
        unexposed_events: 340,
        unexposed_total: 1000,
    };
    let rd = corrected_risk_difference_nondifferential(&table, &rates, 0.1).unwrap();
    assert!(close(rd.naive, 0.14, 1e-12));
    assert!(close(rd.corrected, 0.2, 1e-12));
    assert!(close(rd.risk_exposed, 0.4, 1e-12) && close(rd.risk_unexposed, 0.2, 1e-12));
    let expect_rates = 0.2 * 0.2 * (0.0004 + 0.0009) / 0.49;
    assert!(close(rd.variance_rates, expect_rates, 1e-12));
    // Independent-arm (differential) propagation of the same numbers differs,
    // because the shared rates correlate the arms.
    let split = corrected_risk_difference_differential(&table, &rates, &rates, 0.1).unwrap();
    assert!(close(split.corrected, rd.corrected, 1e-12));
    assert!(!close(split.variance, rd.variance, 1e-9));
}

#[test]
fn differential_misclassification_is_not_recovered_by_pooled_rates() {
    // Exposed arm Se 0.9 / Sp 0.8, unexposed Se 0.7 / Sp 0.95; truth 0.4 vs 0.2.
    // p1* = 0.48, p0* = 0.7*0.2 + 0.05*0.8 = 0.18.
    let e1 = BinaryErrorRates::new(0.9, 0.8, 0.0, 0.0).unwrap();
    let e0 = BinaryErrorRates::new(0.7, 0.95, 0.0, 0.0).unwrap();
    let table = TwoByTwoCounts {
        exposed_events: 480,
        exposed_total: 1000,
        unexposed_events: 180,
        unexposed_total: 1000,
    };
    let diff = corrected_risk_difference_differential(&table, &e1, &e0, 0.1).unwrap();
    assert!(close(diff.corrected, 0.2, 1e-12));
    let pooled = corrected_risk_difference_nondifferential(&table, &e1, 0.1).unwrap();
    // (0.48 - 0.18) / 0.7 = 0.4286: the wrong answer is far from 0.2.
    assert!(close(pooled.corrected, 0.3 / 0.7, 1e-12));
    assert!((pooled.corrected - 0.2).abs() > 0.2);
}

#[test]
fn categorical_inversion_recovers_proportions_and_outcome_means() {
    // True categories 0,1,2 with pi = (0.5, 0.3, 0.2) and mu = (1, 2, 4).
    // Columns are P(observed | true).
    let c = vec![
        0.8, 0.1, 0.0, //
        0.1, 0.8, 0.2, //
        0.1, 0.1, 0.8,
    ];
    let pi = [0.5, 0.3, 0.2];
    let mu = [1.0, 2.0, 4.0];
    let confusion = ConfusionMatrix::new(3, c.clone(), Some(vec![500, 500, 500])).unwrap();
    let n = 10_000.0;
    let mut classes = Vec::new();
    for j in 0..3 {
        let p_star: f64 = (0..3).map(|t| c[j * 3 + t] * pi[t]).sum();
        let ybar: f64 = (0..3).map(|t| c[j * 3 + t] * pi[t] * mu[t]).sum::<f64>() / p_star;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the rounded expected count is a small non-negative integer by construction"
        )]
        classes.push(ObservedClassSummary {
            count: (p_star * n).round() as u64,
            mean: ybar,
            variance: 1.0,
        });
    }
    let counts: Vec<u64> = classes.iter().map(|c| c.count).collect();
    assert_eq!(counts.iter().sum::<u64>(), 10_000);
    let out = corrected_categories(&classes, &confusion, 50.0).unwrap();
    for t in 0..3 {
        assert!(close(out.proportions[t], pi[t], 1e-9), "{:?}", out.proportions);
        assert!(close(out.outcome_means[t], mu[t], 1e-6), "{:?}", out.outcome_means);
    }
    // The naive observed-class means are biased away from mu.
    assert!(!close(classes[2].mean, 4.0, 0.1));
    // Variance is strictly larger than ignoring confusion-matrix uncertainty.
    let exact = ConfusionMatrix::new(3, c, None).unwrap();
    let out_exact = corrected_categories(&classes, &exact, 50.0).unwrap();
    for t in 0..3 {
        assert!(out.proportion_covariance[t * 3 + t] > out_exact.proportion_covariance[t * 3 + t]);
    }
}

#[test]
fn binary_case_of_the_categorical_route_matches_rogan_gladen() {
    // K = 2: C = [[Se, 1-Sp], [1-Se, Sp]], observed positives are class 0.
    let (se, sp) = (0.9, 0.8);
    let confusion =
        ConfusionMatrix::new(2, vec![se, 1.0 - sp, 1.0 - se, sp], Some(vec![200, 300])).unwrap();
    let classes = [
        ObservedClassSummary { count: 410, mean: 1.0, variance: 0.5 },
        ObservedClassSummary { count: 590, mean: 0.0, variance: 0.5 },
    ];
    let out = corrected_categories(&classes, &confusion, 20.0).unwrap();
    let rates =
        BinaryErrorRates::new(se, sp, se * (1.0 - se) / 200.0, sp * (1.0 - sp) / 300.0).unwrap();
    let rg = corrected_prevalence(410, 1000, &rates, 0.1).unwrap();
    assert!(close(out.proportions[0], rg.corrected, 1e-12));
    assert!(close(out.proportion_covariance[0], rg.variance, 1e-10));
}

#[test]
fn ill_conditioned_confusion_refuses() {
    let c = vec![0.5, 0.5, 0.5, 0.5];
    let confusion = ConfusionMatrix::new(2, c, None).unwrap();
    let classes = [
        ObservedClassSummary { count: 50, mean: 0.0, variance: 1.0 },
        ObservedClassSummary { count: 50, mean: 0.0, variance: 1.0 },
    ];
    assert!(matches!(
        corrected_categories(&classes, &confusion, 100.0),
        Err(MeasurementErrorError::ConfusionIllConditioned { .. })
    ));
}

#[test]
fn posterior_mean_route_removes_berkson_noise_that_raw_draws_would_add() {
    // X_i ~ N(0,1); the posterior given data is N(m_i, v) with m_i the
    // posterior mean. Regressing y on a *draw* is attenuated by
    // Var(m) / (Var(m) + v); regressing on the posterior mean is not.
    let n = 4000;
    let draws_count = 40;
    let mut rng = CausalRng::from_seed(7);
    let v: f64 = 0.5;
    let mut means_true = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut draws = vec![0.0; n * draws_count];
    for i in 0..n {
        // latent x = m + sqrt(v) * eps with m ~ N(0, 0.5): Var(x) = 1.
        let m = 0.5_f64.sqrt() * normal(&mut rng);
        let x = m + v.sqrt() * normal(&mut rng);
        means_true.push(m);
        y.push(1.0 + 2.0 * x + normal(&mut rng));
        for s in 0..draws_count {
            draws[s * n + i] = m + v.sqrt() * normal(&mut rng);
        }
    }
    let m_hat = posterior_means(&draws, n, draws_count).unwrap();
    let exact = ErrorVariance::known(0.0, 0.0).unwrap();
    let on_mean = LinearMoments::accumulate(
        &LinearErrorData {
            y: &y,
            w: &m_hat,
            covariates: &[],
            covariate_count: 0,
            replicate_counts: None,
        },
        &exact,
        0.0,
    )
    .unwrap()
    .solve()
    .unwrap();
    // Truth 2.0; the posterior-mean MC error adds v/S noise to m (small).
    assert!(close(
        on_mean.naive_coefficients[1],
        2.0,
        4.0 * on_mean.naive_covariance[3].sqrt() + 0.05
    ));
    let on_draw = LinearMoments::accumulate(
        &LinearErrorData {
            y: &y,
            w: &draws[..n],
            covariates: &[],
            covariate_count: 0,
            replicate_counts: None,
        },
        &exact,
        0.0,
    )
    .unwrap()
    .solve()
    .unwrap();
    // Attenuation factor 0.5 / (0.5 + 0.5) = 0.5 -> about 1.0.
    assert!(on_draw.naive_coefficients[1] < 1.2, "{}", on_draw.naive_coefficients[1]);
}

#[test]
fn rubin_combination_has_between_imputation_component() {
    // Three exact imputations with slopes differing across draws.
    let n = 60;
    let y: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1 + ((i * 7) % 5) as f64 * 0.05).collect();
    let base: Vec<f64> = (0..n).map(|i| ((i * 3) % 11) as f64 * 0.2 + i as f64 * 0.1).collect();
    let mut draws = Vec::new();
    for s in 0..3 {
        for (i, &b) in base.iter().enumerate() {
            draws.push(b + 0.1 * s as f64 * (((i * 5 + s) % 7) as f64 - 3.0));
        }
    }
    let combined = combine_imputations_linear(&y, &draws, 3, &[], 0).unwrap();
    assert_eq!(combined.draws, 3);
    let total = combined.covariance[3];
    let expect = combined.within[3] + (1.0 + 1.0 / 3.0) * combined.between[3];
    assert!(close(total, expect, 1e-15));
    assert!(combined.between[3] > 0.0);
    assert!(combine_imputations_linear(&y, &draws[..n], 1, &[], 0).is_err());
}

#[test]
fn validation_sample_estimates_the_error_variance() {
    // W - X alternates +0.5 / -0.5 around zero: mean 0, variance 0.25 * n/(n-1).
    let x: Vec<f64> = (0..10).map(f64::from).collect();
    let w: Vec<f64> =
        x.iter().enumerate().map(|(i, v)| v + if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
    let (err, summary) = ErrorVariance::from_validation(&w, &x).unwrap();
    assert!(close(summary.error_mean, 0.0, 1e-15));
    assert!(close(err.variance, 0.25 * 10.0 / 9.0, 1e-12));
    assert!(err.influence.is_none());
}

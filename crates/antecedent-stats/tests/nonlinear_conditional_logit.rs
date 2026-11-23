//! Conditional logit with shape parameters: the linear case against the ordinary fit, recovery
//! of a simulated model, the profile against an independent one-dimensional maximisation, and
//! every identification refusal.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{
    ChoiceUtility, ConditionalLogitDesign, ConditionalLogitOptions, NonlinearChoiceError,
    NonlinearConditionalLogitOptions, ShapeSearchOptions, UtilityNotEvaluable,
    conditional_logit_log_likelihood, fit_conditional_logit, fit_nonlinear_conditional_logit,
};

/// A deterministic uniform generator.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        f64::from(u32::try_from(self.0 >> 40).unwrap()) / f64::from(1_u32 << 24)
    }

    fn between(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// How the value of an option reads its two attributes `(a, b)` and the shape parameter `s`.
#[derive(Clone, Copy)]
enum Form {
    /// `F = a`, `T = b`; no shape parameter.
    Linear,
    /// `F = -exp(-s a)`; no linear parameter.
    Exponential,
    /// `F = a + b exp(-(s - 1)^2)`; no linear parameter.
    Bump,
    /// Four shape parameters, each `F = a s_k`; for the search limit only.
    FourShapes,
}

/// Two-option choice sets with attributes `(a, b)` per option.
struct Sets {
    form: Form,
    rows: Vec<(f64, f64)>,
    starts: Vec<usize>,
    chosen: Vec<usize>,
}

impl Sets {
    fn value(&self, row: (f64, f64), shape: &[f64]) -> f64 {
        match self.form {
            Form::Linear => row.0,
            Form::Exponential => -(-shape[0] * row.0).exp(),
            Form::Bump => {
                let d = shape[0] - 1.0;
                row.0 + row.1 * (-d * d).exp()
            }
            Form::FourShapes => row.0 * shape.iter().sum::<f64>(),
        }
    }

    /// Simulate `n` sets: attributes drawn by `draw`, the first option chosen with logit
    /// probability at consistency `lambda` and shape `truth`.
    fn simulate(
        form: Form,
        n: usize,
        seed: u64,
        lambda: f64,
        truth: &[f64],
        draw: impl Fn(&mut Lcg) -> (f64, f64),
    ) -> Self {
        let mut rng = Lcg(seed);
        let mut sets = Self { form, rows: Vec::new(), starts: vec![0], chosen: Vec::new() };
        for _ in 0..n {
            let first = draw(&mut rng);
            let second = draw(&mut rng);
            let gap = lambda * (sets.value(first, truth) - sets.value(second, truth));
            let p_first = 1.0 / (1.0 + (-gap).exp());
            sets.chosen.push(usize::from(rng.unit() >= p_first));
            sets.rows.push(first);
            sets.rows.push(second);
            sets.starts.push(sets.rows.len());
        }
        sets
    }
}

impl ChoiceUtility for Sets {
    fn n_linear(&self) -> usize {
        usize::from(matches!(self.form, Form::Linear))
    }

    fn n_shape(&self) -> usize {
        match self.form {
            Form::Linear => 0,
            Form::Exponential | Form::Bump => 1,
            Form::FourShapes => 4,
        }
    }

    fn set_starts(&self) -> &[usize] {
        &self.starts
    }

    fn chosen(&self) -> &[usize] {
        &self.chosen
    }

    fn clusters(&self) -> Option<&[u32]> {
        None
    }

    fn features(&self, shape: &[f64]) -> Result<Vec<f64>, UtilityNotEvaluable> {
        Ok(match self.form {
            Form::Linear => self.rows.iter().flat_map(|r| [r.0, r.1]).collect(),
            _ => self.rows.iter().map(|r| self.value(*r, shape)).collect(),
        })
    }

    fn shape_jacobian(&self, shape: &[f64]) -> Result<Vec<f64>, UtilityNotEvaluable> {
        Ok(match self.form {
            Form::Linear => Vec::new(),
            Form::Exponential => self.rows.iter().map(|r| r.0 * (-shape[0] * r.0).exp()).collect(),
            Form::Bump => {
                let d = shape[0] - 1.0;
                self.rows.iter().map(|r| -2.0 * d * r.1 * (-d * d).exp()).collect()
            }
            Form::FourShapes => self.rows.iter().flat_map(|r| [r.0; 4]).collect(),
        })
    }
}

fn options() -> NonlinearConditionalLogitOptions {
    NonlinearConditionalLogitOptions::default()
}

#[test]
fn without_shape_parameters_the_fit_is_the_ordinary_conditional_logit() {
    let sets = Sets::simulate(Form::Linear, 300, 11, 1.0, &[], |rng| {
        (rng.between(-1.0, 2.0), rng.between(-1.5, 1.5))
    });
    let features: Vec<f64> = sets.rows.iter().flat_map(|r| [r.0, 0.7 * r.0 - 1.3 * r.1]).collect();
    // Re-simulate choices from a known beta so that both columns matter.
    let mut sets = sets;
    let mut rng = Lcg(5);
    for (s, chosen) in sets.chosen.iter_mut().enumerate() {
        let gap = features[4 * s + 1] - features[4 * s + 3];
        *chosen = usize::from(rng.unit() >= 1.0 / (1.0 + (-gap).exp()));
    }
    let raw: Vec<f64> = sets.rows.iter().flat_map(|r| [r.0, r.1]).collect();
    let design = ConditionalLogitDesign {
        features: &raw,
        n_features: 2,
        set_starts: &sets.starts,
        chosen: &sets.chosen,
        clusters: None,
    };
    let ordinary = fit_conditional_logit(&design, ConditionalLogitOptions::default()).unwrap();
    let fit = fit_nonlinear_conditional_logit(&sets, &[], options(), None).unwrap();
    let beta = &ordinary.coefficients;
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-10 * (1.0 + a.abs().max(b.abs()));
    assert!(close(fit.consistency, beta[0]), "{} {}", fit.consistency, beta[0]);
    assert!(close(fit.linear[0], beta[1] / beta[0]));
    assert!(close(fit.log_likelihood, ordinary.log_likelihood));
    assert!(close(fit.null_log_likelihood, ordinary.null_log_likelihood));
    assert!(fit.shape.is_empty() && fit.profiles.is_empty());
    // The covariance of (lambda, theta) is the delta-method transform of beta's covariance.
    let v = &ordinary.covariance;
    let theta = beta[1] / beta[0];
    let var_theta = (v[3] - 2.0 * theta * v[2] + theta * theta * v[0]) / (beta[0] * beta[0]);
    assert!(close(fit.covariance[0], v[0]));
    assert!(close(fit.covariance[3], var_theta), "{} {}", fit.covariance[3], var_theta);
}

/// The profile at `s` by golden-section search over the consistency alone: the log-likelihood is
/// concave in it.
fn golden_profile(sets: &Sets, s: f64) -> f64 {
    let features = sets.features(&[s]).unwrap();
    let design = ConditionalLogitDesign {
        features: &features,
        n_features: 1,
        set_starts: &sets.starts,
        chosen: &sets.chosen,
        clusters: None,
    };
    let at = |lambda: f64| conditional_logit_log_likelihood(&design, &[lambda]).unwrap();
    let (mut lo, mut hi) = (-50.0_f64, 400.0_f64);
    let ratio = 0.5 * (5.0_f64.sqrt() - 1.0);
    for _ in 0..200 {
        let left = hi - ratio * (hi - lo);
        let right = lo + ratio * (hi - lo);
        if at(left) < at(right) {
            lo = left;
        } else {
            hi = right;
        }
    }
    at(0.5 * lo + 0.5 * hi)
}

#[test]
fn a_simulated_risk_attitude_is_recovered_and_its_profile_is_the_true_profile() {
    let sets =
        Sets::simulate(Form::Exponential, 400, 7, 6.0, &[0.7], |rng| (rng.between(0.0, 4.0), 0.0));
    let fit = fit_nonlinear_conditional_logit(&sets, &[(0.05, 3.0)], options(), None).unwrap();
    assert!((fit.shape[0] - 0.7).abs() < 0.25, "shape {}", fit.shape[0]);
    assert!(fit.consistency > 3.0 && fit.consistency < 12.0, "lambda {}", fit.consistency);
    let profile = &fit.profiles[0];
    assert!(profile.interval.0 < fit.shape[0] && fit.shape[0] < profile.interval.1);
    assert!(!profile.lower_truncated && !profile.upper_truncated);
    // Every profile value is the maximum over the consistency, found independently.
    for (s, value) in profile.values.iter().zip(&profile.log_likelihood) {
        let brute = golden_profile(&sets, *s);
        assert!((brute - value).abs() <= 1e-6 * (1.0 + value.abs()), "s {s}: {brute} vs {value}");
    }
    // The estimate is the maximum of the independent profile on a fine grid.
    let best = (0..=600)
        .map(|i| 0.05 + 2.95 * f64::from(i) / 600.0)
        .map(|s| (s, golden_profile(&sets, s)))
        .fold((0.0, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
    assert!((best.0 - fit.shape[0]).abs() <= 2.0 * 2.95 / 600.0, "{best:?} vs {}", fit.shape[0]);
    assert!(fit.log_likelihood >= best.1 - 1e-9);
    // The likelihood-ratio interval ends sit where the profile falls by chi2_1(0.95) / 2.
    let cutoff = fit.log_likelihood - 0.5 * 1.959_963_984_540_054_f64.powi(2);
    for end in [profile.interval.0, profile.interval.1] {
        assert!((golden_profile(&sets, end) - cutoff).abs() <= 1e-4, "end {end}");
    }
    let q = 2;
    assert!(fit.covariance[0] > 0.0 && fit.covariance[q * q - 1] > 0.0);
}

#[test]
fn two_equal_profile_maxima_with_a_deep_dip_between_are_refused() {
    let truth = 1.0 - 2.0_f64.ln().sqrt();
    let sets = Sets::simulate(Form::Bump, 1500, 3, 5.0, &[truth], |rng| {
        (rng.between(-1.0, 1.0), rng.between(-2.0, 2.0))
    });
    let error = fit_nonlinear_conditional_logit(&sets, &[(0.0, 2.0)], options(), None).unwrap_err();
    let NonlinearChoiceError::ProfileMultimodal { dimension, modes } = error else {
        panic!("{error:?}");
    };
    assert_eq!(dimension, 0);
    assert!((modes.0 + modes.1 - 2.0).abs() < 0.2, "{modes:?}");
    assert!(modes.0 < 0.6 && modes.1 > 1.4, "{modes:?}");
}

#[test]
fn a_shape_the_choices_barely_inform_is_refused() {
    let sets = Sets::simulate(Form::Exponential, 300, 17, 2000.0, &[0.05], |rng| {
        (rng.between(0.0, 0.02), 0.0)
    });
    let error =
        fit_nonlinear_conditional_logit(&sets, &[(0.05, 2.0)], options(), None).unwrap_err();
    assert!(
        matches!(error, NonlinearChoiceError::ShapeNotIdentified { dimension: 0, profile_range }
            if profile_range < 0.5 * 1.959_963_984_540_054_f64.powi(2)),
        "{error:?}"
    );
    assert_eq!(error.detail(), "nonlinear_conditional_logit.shape_not_identified");
}

#[test]
fn an_estimate_on_a_declared_bound_is_refused_with_its_truncated_interval() {
    // Risk-neutral choices read with a risk attitude of at least 0.3: the profile falls from the
    // lower bound.
    let mut sets =
        Sets::simulate(Form::Exponential, 800, 23, 1.0, &[0.3], |rng| (rng.between(0.0, 5.0), 0.0));
    let mut rng = Lcg(29);
    for (s, chosen) in sets.chosen.iter_mut().enumerate() {
        let gap = 2.0 * (sets.rows[2 * s].0 - sets.rows[2 * s + 1].0);
        *chosen = usize::from(rng.unit() >= 1.0 / (1.0 + (-gap).exp()));
    }
    let error = fit_nonlinear_conditional_logit(&sets, &[(0.3, 2.0)], options(), None).unwrap_err();
    let NonlinearChoiceError::ShapeEstimateAtBound { dimension, estimate, interval } = error else {
        panic!("{error:?}");
    };
    assert_eq!(dimension, 0);
    assert!((estimate - 0.3).abs() <= 1e-6, "{estimate}");
    assert!(interval.0.total_cmp(&0.3).is_eq() && interval.1 > 0.3 && interval.1 < 2.0);
}

#[test]
fn separated_choices_are_refused_naming_the_shape_point() {
    let mut sets =
        Sets::simulate(Form::Exponential, 50, 31, 1.0, &[1.0], |rng| (rng.between(0.0, 4.0), 0.0));
    for (s, chosen) in sets.chosen.iter_mut().enumerate() {
        *chosen = usize::from(sets.rows[2 * s + 1].0 > sets.rows[2 * s].0);
    }
    let error = fit_nonlinear_conditional_logit(&sets, &[(0.1, 2.0)], options(), None).unwrap_err();
    assert!(matches!(error, NonlinearChoiceError::Separated { ref shape } if shape.len() == 1));
}

#[test]
fn controls_bounds_and_the_search_limit_are_checked() {
    let sets = Sets::simulate(Form::FourShapes, 20, 37, 1.0, &[0.1; 4], |rng| {
        (rng.between(0.0, 1.0), 0.0)
    });
    let bounds = [(0.0, 1.0); 4];
    assert_eq!(
        fit_nonlinear_conditional_logit(&sets, &bounds, options(), None).unwrap_err(),
        NonlinearChoiceError::ShapeSearchBudget { dimensions: 4, limit: 3 }
    );
    let one =
        Sets::simulate(Form::Exponential, 20, 41, 1.0, &[0.5], |rng| (rng.between(0.0, 1.0), 0.0));
    assert_eq!(
        fit_nonlinear_conditional_logit(&one, &[(1.0, 0.5)], options(), None).unwrap_err(),
        NonlinearChoiceError::BoundsInvalid { dimension: 0 }
    );
    assert_eq!(
        fit_nonlinear_conditional_logit(&one, &[], options(), None).unwrap_err(),
        NonlinearChoiceError::BoundsInvalid { dimension: 0 }
    );
    let level = NonlinearConditionalLogitOptions { level: 1.5, ..options() };
    assert_eq!(
        fit_nonlinear_conditional_logit(&one, &[(0.1, 1.0)], level, None).unwrap_err(),
        NonlinearChoiceError::InvalidOptions
    );
    let search = NonlinearConditionalLogitOptions {
        search: ShapeSearchOptions { profile_points: 2, ..ShapeSearchOptions::default() },
        ..options()
    };
    assert_eq!(
        fit_nonlinear_conditional_logit(&one, &[(0.1, 1.0)], search, None).unwrap_err(),
        NonlinearChoiceError::InvalidOptions
    );
}

use antecedent_core::{
    Assumption, AssumptionSet, CausalResponse, ContinuousDomain, DerivativeScale,
    DerivativeWeighting, GridSpec, IdentificationStatus, Intervention, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue, StochasticPolicy,
    SupportRegion, SupportReport, SupportStatus, Value, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_stats::{
    GamWorkspace, LocalQuadraticWorkspace, gaussian_density,
    gaussian_local_quadratic_influence_prechecked, normal_ppf,
};

use super::bayes::{inflate_draws_by_edf, summarize_scalar_draws};
use super::plugin::predict_one;
use super::sample::CompleteSample;
use super::*;

/// The draws a Bayesian response retains on its credible uncertainty are
/// exactly the vector its endpoints are the quantiles of: re-summarizing
/// every coordinate at the published level under the estimator's own rule
/// reproduces `[lower, upper]` bit for bit.
fn assert_retained_draws_publish(uncertainty: &ResponseUncertainty, n_draws: usize) {
    let (level, published): (f64, Vec<(f64, f64)>) = match uncertainty {
        ResponseUncertainty::Scalar { level, lower, upper, .. } => (*level, vec![(*lower, *upper)]),
        ResponseUncertainty::PointwiseBand { level, lower, upper, .. } => {
            (*level, lower.iter().copied().zip(upper.iter().copied()).collect())
        }
        other => panic!("expected a credible interval or band, got {other:?}"),
    };
    let draws = uncertainty.credible_draws().expect("credible uncertainty retains its draws");
    assert_eq!(draws.n_draws, n_draws);
    assert_eq!(draws.n_coordinates(), published.len());
    for (j, (lo, hi)) in published.into_iter().enumerate() {
        let (_, rebuilt_lo, rebuilt_hi, _) =
            summarize_scalar_draws(draws.column(j).unwrap(), level).unwrap();
        assert!(
            rebuilt_lo.to_bits() == lo.to_bits() && rebuilt_hi.to_bits() == hi.to_bits(),
            "coordinate {j}: [{lo}, {hi}] != retained-draw quantiles [{rebuilt_lo}, {rebuilt_hi}]"
        );
    }
}

#[test]
fn bayesian_responses_retain_the_draws_their_bands_summarize() {
    let (data, a, y, x) = confounded_curve(240);
    let estimator = ContinuousResponseEstimator::new([x]);
    let bayes = crate::BayesianGComputationAte::new().with_n_draws(8);
    let ctx = antecedent_core::ExecutionContext::for_tests(18);
    let run = |functional| {
        estimator
            .estimate_bayesian(
                &data,
                &ResponseQuery::new(functional),
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &bayes,
                &ctx,
            )
            .unwrap()
    };
    // Linear route: one column per grid point of the curve.
    let curve = run(ResponseFunctional::MeanCurve {
        outcome: y,
        treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.5, 0.0, 0.5]))),
    });
    assert_retained_draws_publish(&curve.uncertainty, bayes.n_draws);
    assert_eq!(curve.uncertainty.credible_draws().unwrap().n_coordinates(), 3);
    // Linear route, scalar.
    let level = run(ResponseFunctional::InterventionResponse {
        outcome: y,
        interventions: Arc::from([Intervention::set(a, Value::f64(0.25))]),
    });
    assert_retained_draws_publish(&level.uncertainty, bayes.n_draws);
    // Additive-GAM plug-in band: retained *after* the edf spread inflation.
    let jacobian = run(ResponseFunctional::Jacobian {
        outcomes: Arc::from([y]),
        treatments: Arc::from([a]),
        at: Arc::from([0.2]),
        scale: DerivativeScale::Identity,
    });
    assert_retained_draws_publish(&jacobian.uncertainty, bayes.n_draws);
}

#[test]
fn bayesian_logit_response_evaluates_each_draw_on_probability_scale() {
    let n = 240;
    let mut treatment = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    let mut covariate = Vec::with_capacity(n);
    for row in 0..n {
        let x = -1.0 + 2.0 * row as f64 / (n - 1) as f64;
        let a = -0.8 + 1.6 * ((row * 53 % n) as f64 / n as f64);
        let p = antecedent_stats::GlmFamily::BinomialLogit.mean_from_eta(-0.4 + 0.8 * a + 0.7 * x);
        let y = f64::from((row * 37 % 101) as f64 / 101.0 < p);
        covariate.push(x);
        treatment.push(a);
        outcome.push(y);
    }
    let data = TabularData::from_f64_columns([
        ("a", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("x", covariate.as_slice()),
    ])
    .unwrap();
    let a = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let x = VariableId::from_raw(2);
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: y,
        treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.4, 0.0, 0.4]))),
    });
    let bayes = crate::BayesianGComputationAte::new()
        .with_likelihood(antecedent_prob::BayesLikelihood::BernoulliLogit)
        .with_n_draws(80)
        .with_seed(31);
    let result = ContinuousResponseEstimator::new([x])
        .estimate_bayesian(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
            &bayes,
            &antecedent_core::ExecutionContext::for_tests(31),
        )
        .unwrap();
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &result.estimate
    else {
        panic!("expected response surface")
    };
    assert!(mean.iter().all(|value| (0.0..=1.0).contains(value)));
    assert!(mean[0] < mean[2]);
    assert_retained_draws_publish(&result.uncertainty, 80);
}

#[test]
fn bayesian_curve_simultaneous_band_uses_joint_posterior_draws() {
    let (data, a, y, x) = confounded_curve(240);
    let bayes = crate::BayesianGComputationAte::new().with_n_draws(200).with_seed(31);
    let ctx = antecedent_core::ExecutionContext::for_tests(18);
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: y,
        treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.5, 0.0, 0.5]))),
    });
    let mut pointwise = ContinuousResponseEstimator::new([x]);
    pointwise.options.bandwidth = Some(0.4);
    let ordinary = pointwise
        .estimate_bayesian(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
            &bayes,
            &ctx,
        )
        .unwrap();
    pointwise.options.simultaneous_replicates = Some(100);
    let joint = pointwise
        .estimate_bayesian(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
            &bayes,
            &ctx,
        )
        .unwrap();
    let ResponseUncertainty::PointwiseBand { lower: p_lo, upper: p_hi, .. } = ordinary.uncertainty
    else {
        panic!("expected pointwise credible band");
    };
    let ResponseUncertainty::SimultaneousBand {
        lower: s_lo,
        upper: s_hi,
        replicates,
        interpretation,
        ..
    } = joint.uncertainty
    else {
        panic!("expected simultaneous credible band");
    };
    assert_eq!(replicates, 200);
    assert_eq!(interpretation, antecedent_core::IntervalInterpretation::Credible);
    for ((&lo, &hi), (&point_lo, &point_hi)) in
        s_lo.iter().zip(s_hi.iter()).zip(p_lo.iter().zip(p_hi.iter()))
    {
        assert!(lo <= point_lo && hi >= point_hi);
    }
}

/// Minimal deterministic uniform stream (`SplitMix64`) for exchangeability checks.
fn uniform_stream(mut state: u64) -> impl FnMut() -> f64 {
    move || crate::splitmix::splitmix64_unit(&mut state)
}

#[test]
fn exchangeable_rank_interval_covers_an_exchangeable_truth_at_the_level() {
    // With the truth exchangeable with the draws, the type-6 interval
    // covers at the level for any draw count; the type-7 interval of 20
    // draws covers about 0.81 at a 90% level.
    let mut u = uniform_stream(0x5EED);
    let trials = 20_000;
    for draws_n in [20usize, 64, 200] {
        let (mut covered, mut covered_type7) = (0u32, 0u32);
        for _ in 0..trials {
            let truth = u();
            let draws: Vec<f64> = (0..draws_n).map(|_| u()).collect();
            let (_, lo, hi, _) = summarize_scalar_draws(&draws, 0.9).unwrap();
            covered += u32::from(truth >= lo && truth <= hi);
            let mut sorted = draws;
            sorted.sort_by(f64::total_cmp);
            let type7 = |p: f64| {
                let x = p * (sorted.len() - 1) as f64;
                let (l, h) = (x.floor() as usize, x.ceil() as usize);
                sorted[l] + (sorted[h] - sorted[l]) * (x - l as f64)
            };
            covered_type7 += u32::from(truth >= type7(0.05) && truth <= type7(0.95));
        }
        let rate = f64::from(covered) / f64::from(trials);
        assert!((rate - 0.9).abs() < 0.01, "draws={draws_n}: type-6 coverage {rate}");
        if draws_n == 20 {
            let rate7 = f64::from(covered_type7) / f64::from(trials);
            assert!(rate7 < 0.84, "draws=20: type-7 coverage {rate7} should be ≈ 0.81");
        }
    }
}

#[test]
fn inflate_draws_by_edf_scales_spread_and_keeps_mean() {
    let mut draws = vec![1.0, 2.0, 4.0, 5.0];
    inflate_draws_by_edf(&mut draws, 100, 20.0);
    let factor = (100.0_f64 / 80.0).sqrt();
    let expect = |v: f64| 3.0 + (v - 3.0) * factor;
    for (got, raw) in draws.iter().zip([1.0, 2.0, 4.0, 5.0]) {
        assert!((got - expect(raw)).abs() < 1e-12);
    }
    assert!((draws.iter().sum::<f64>() / 4.0 - 3.0).abs() < 1e-12);
    // Degenerate edf (≥ n or non-finite) leaves the draws alone.
    for edf in [100.0, 150.0, f64::NAN, -1.0] {
        let mut untouched = vec![1.0, 2.0, 4.0, 5.0];
        inflate_draws_by_edf(&mut untouched, 100, edf);
        assert_eq!(untouched, vec![1.0, 2.0, 4.0, 5.0], "edf={edf}");
    }
}

#[test]
fn bayesian_point_derivative_refits_nuisances_per_draw() {
    let (data, a, y, x) = confounded_curve(240);
    let mut estimator = ContinuousResponseEstimator::new([x]);
    estimator.options.bandwidth = Some(0.35);
    let sample = CompleteSample::read(&data, y, &[a], &[x]).unwrap();
    let frozen = estimator.cross_fitted_pseudo_outcome(&sample).unwrap().values;
    let folds = estimator.cross_fit_folds(&sample).unwrap();
    let ones = vec![1.0; sample.len()];
    let unit =
        estimator.cross_fitted_pseudo_outcome_weighted(&sample, &folds, Some(&ones)).unwrap();
    let unit_gap = frozen.iter().zip(&unit.values).map(|(f, u)| (f - u).abs()).fold(0.0, f64::max);
    assert!(unit_gap < 1e-9, "unit weights must reproduce the unweighted pseudo-outcome");
    let weights: Vec<f64> = (0..sample.len()).map(|i| if i % 3 == 0 { 2.2 } else { 0.4 }).collect();
    let refit =
        estimator.cross_fitted_pseudo_outcome_weighted(&sample, &folds, Some(&weights)).unwrap();
    let refit_gap =
        frozen.iter().zip(&refit.values).map(|(f, r)| (f - r).abs()).fold(0.0, f64::max);
    assert!(refit_gap > 1e-6, "weighted draws must refit the nuisance stage, not reuse φ");

    let bayes = crate::BayesianGComputationAte::new().with_n_draws(8);
    for scale in [DerivativeScale::Identity, DerivativeScale::LogLog] {
        let query = ResponseQuery::new(ResponseFunctional::PointDerivative {
            outcome: y,
            treatment: a,
            at: 0.2,
            order: 1,
            scale,
        });
        let response = estimator
            .estimate_bayesian(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &bayes,
                &antecedent_core::ExecutionContext::for_tests(18),
            )
            .unwrap();
        assert_eq!(response.provenance_id.as_ref(), "estimate.response.point_derivative");
        // Posterior draws: the "standard error" is a posterior SD and the interval is
        // credible, and the tag says so.
        assert!(matches!(
            response.uncertainty,
            ResponseUncertainty::Scalar {
                interpretation: antecedent_core::IntervalInterpretation::Credible,
                ..
            }
        ));
        // The retained draws are the bias-corrected ones the interval is read from.
        assert_retained_draws_publish(&response.uncertainty, 8);
        assert!(response.assumptions.entries.iter().any(|record| match &record.assumption {
            Assumption::ParametricRestriction(restriction) =>
                restriction.description.contains("Not a frozen-pseudo-outcome reweight"),
            _ => false,
        }));
        let codes: Vec<_> =
            response.support.warnings.iter().map(|w| w.code.as_ref().to_owned()).collect();
        assert!(
            codes.iter().any(|c| c == "response.derivative_interval_bias_corrected"),
            "{codes:?}"
        );
        assert!(!codes.iter().any(|c| c == "response.derivative_interval_withheld"));
        let note = response
            .support
            .warnings
            .iter()
            .find(|w| w.code.as_ref() == "response.derivative_interval_bias_corrected")
            .unwrap();
        assert!(note.message.contains("refits the cross-fitted"), "{}", note.message);
    }
}

fn confounded_curve(n: usize) -> (TabularData, VariableId, VariableId, VariableId) {
    let mut a = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        let z = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
        let noise = ((i * 37 % 101) as f64 / 100.0 - 0.5) * 0.3;
        let treatment = 0.7 * z + noise;
        x.push(z);
        a.push(treatment);
        y.push(1.0 + 2.0 * treatment + 0.8 * z + 0.05 * (i as f64).sin());
    }
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("y", y.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap();
    (data, VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2))
}

#[test]
fn dr_curve_calibrates_linear_response_and_reports_support() {
    let (data, a, y, x) = confounded_curve(500);
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: y,
        treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.4, 0.0, 0.4]))),
    });
    let response = ContinuousResponseEstimator::new([x])
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .unwrap();
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        response.estimate
    else {
        panic!("expected surface");
    };
    // y = 1 + 2a + 0.8x + tiny with E[x] = 0, so E[y | do(a)] = 1 + 2a; the unadjusted
    // regression slope would be about 2.8. Observed smoothing error is at most 0.045.
    for (got, level) in mean.iter().zip([-0.4, 0.0, 0.4]) {
        assert!((got - (1.0 + 2.0 * level)).abs() < 0.07, "means={mean:?}");
    }
    assert!((mean[2] - mean[0] - 1.6).abs() < 0.15, "means={mean:?}");
    assert_ne!(response.support.status, SupportStatus::OutsideEmpiricalSupport);
    assert_eq!(response.support.diagnostics[0].values.len(), 3);
    assert_eq!(response.provenance_id.as_ref(), "estimate.response.kennedy_dr");
    assert!(response.assumptions.entries.iter().any(|record| matches!(
        &record.assumption,
        Assumption::ParametricRestriction(restriction)
            if restriction.id.as_ref() == "response.kennedy_dr.nuisance_regularity"
    )));
    assert!(
        response
            .support
            .warnings
            .iter()
            .any(|d| d.code.as_ref() == "response.kennedy_dr.additive_covariate_if"),
        "Kennedy IF covariate term must disclose the additive-μ restriction"
    );
    let tail = response
        .support
        .diagnostics
        .iter()
        .find(|d| d.id.as_ref() == "response.outcome_tail_ratio")
        .expect("outcome tail ratio");
    assert_eq!(tail.values.len(), 2);
    assert!(tail.values[0] < OUTCOME_TAIL_RATIO_BOUND, "ratio={}", tail.values[0]);
    assert!((tail.values[1] - OUTCOME_TAIL_RATIO_BOUND).abs() < 1e-12);
    assert!(
        response
            .support
            .warnings
            .iter()
            .all(|w| w.code.as_ref() != "response.heavy_tailed_outcome")
    );
    let winsor = response
        .support
        .diagnostics
        .iter()
        .find(|d| d.id.as_ref() == "response.pseudo_outcome_winsor_shift")
        .expect("winsor shift");
    assert_eq!(winsor.values.len(), 3);
}

#[test]
fn kennedy_curve_discloses_additive_if_on_a_treatment_covariate_product() {
    let n = 80usize;
    let a: Vec<f64> = (0..n).map(|i| (i as f64 / (n - 1) as f64) * 2.0 - 1.0).collect();
    let x: Vec<f64> = (0..n).map(|i| ((i * 17) % 11) as f64 / 5.0 - 1.0).collect();
    let y: Vec<f64> = a.iter().zip(&x).map(|(a, x)| a * x).collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("y", y.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([-0.5, 0.0, 0.5])),
        ),
    });
    let response = ContinuousResponseEstimator::new([VariableId::from_raw(2)])
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .unwrap();
    assert!(
        response
            .support
            .warnings
            .iter()
            .any(|d| d.code.as_ref() == "response.kennedy_dr.additive_covariate_if")
    );
    assert!(matches!(
        response.estimate,
        ResponseIdentification::PointIdentified(ResponseValue::Surface { .. })
    ));
}

fn overwrite_outcome(data: &TabularData, patches: &[(usize, f64)]) -> TabularData {
    let mut columns: Vec<Vec<f64>> =
        (0..3).map(|c| data.float64_values(VariableId::from_raw(c)).unwrap()).collect();
    for &(i, value) in patches {
        columns[1][i] = value;
    }
    TabularData::from_f64_columns([
        ("a", columns[0].as_slice()),
        ("y", columns[1].as_slice()),
        ("x", columns[2].as_slice()),
    ])
    .unwrap()
}

#[test]
fn outcome_tail_ratio_is_zero_on_a_constant_and_explodes_on_a_spike() {
    assert!(outcome_tail_ratio(&[1.0, 1.0, 1.0, 1.0]).abs() < 1e-12);
    let mut y = vec![0.0; 21];
    y[0] = 500.0;
    assert!(outcome_tail_ratio(&y) > OUTCOME_TAIL_RATIO_BOUND);
}

#[test]
fn heavy_tailed_outcome_warns_without_demoting_overlap() {
    let (data, a, y, x) = confounded_curve(240);
    let data = overwrite_outcome(&data, &[(120, 1_000.0), (121, -1_000.0)]);
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: y,
        treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.4, 0.0, 0.4]))),
    });
    let response = ContinuousResponseEstimator::new([x])
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .unwrap();
    assert_eq!(response.support.status, SupportStatus::Supported);
    let tail = response
        .support
        .diagnostics
        .iter()
        .find(|d| d.id.as_ref() == "response.outcome_tail_ratio")
        .expect("outcome tail ratio");
    assert!(tail.values[0] > OUTCOME_TAIL_RATIO_BOUND, "ratio={}", tail.values[0]);
    let codes: Vec<&str> = response.support.warnings.iter().map(|w| w.code.as_ref()).collect();
    assert!(codes.contains(&"response.heavy_tailed_outcome"), "codes={codes:?}");
    let winsor = response
        .support
        .diagnostics
        .iter()
        .find(|d| d.id.as_ref() == "response.pseudo_outcome_winsor_shift")
        .expect("winsor shift");
    assert_eq!(winsor.values.len(), 3);
    assert!(winsor.values.iter().all(|v| v.is_finite()));
}

#[test]
fn pseudo_outcome_winsor_shift_warns_when_a_spike_moves_the_curve() {
    let n = 80usize;
    let treatments: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let mut phi: Vec<f64> = treatments.iter().map(|t| 2.0 * t).collect();
    phi[n / 2] = 400.0;
    let grid = [0.25, 0.5, 0.75];
    let mut workspace = LocalQuadraticWorkspace::default();
    let mean: Vec<f64> = grid
        .iter()
        .map(|&at| {
            gaussian_local_quadratic_influence_prechecked(
                &mut workspace,
                &treatments,
                &phi,
                at,
                0.15,
            )
            .unwrap()
            .point
            .value
        })
        .collect();
    let mut support = SupportReport {
        status: SupportStatus::Supported,
        query_region: SupportRegion { minima: Arc::from([0.0]), maxima: Arc::from([1.0]) },
        diagnostics: Vec::new(),
        warnings: Vec::new(),
        point_status: None,
    };
    push_pseudo_outcome_winsor_shift(
        &mut support,
        &treatments,
        &phi,
        &grid,
        &mean,
        0.15,
        &mut workspace,
    );
    assert!(
        support
            .warnings
            .iter()
            .any(|w| w.code.as_ref() == "response.pseudo_outcome_tail_sensitivity"),
        "codes={:?}",
        support.warnings.iter().map(|w| w.code.as_ref()).collect::<Vec<_>>()
    );
}

#[test]
fn pseudo_outcome_additive_hoist_matches_brute_force_double_loop() {
    // The O(n) covariate-offset form must agree with the definitional
    // |valid|×n double loop (full counterfactual prediction per pair)
    // up to floating-point re-association. 160 rows sums the marginal density
    // term by term; 900 rows (720 training rows per fold) takes the binned
    // Gauss-transform path, which must agree with the same definitional sum.
    //
    // The covariate marginalization `∫μ̂(a, x) dP_n(x)` (Kennedy et al. 2017,
    // Thm. 3) is the empirical mean over the *full* sample, so `marginal_mu`
    // below averages the counterfactual prediction over every row `j` in
    // `0..n`, not only the fold's training rows — matching
    // `cross_fitted_pseudo_outcome_weighted`'s `covariate_offset`. The
    // marginal *density* mixture stays a training-row-only estimate, matching
    // the production `marginal_mixture`, which this test does not change.
    for rows in [160, 900] {
        let (data, a, y, x) = confounded_curve(rows);
        let estimator = ContinuousResponseEstimator::new([x]);
        let sample = CompleteSample::read(&data, y, &[a], &estimator.adjustment_set).unwrap();
        let fast = estimator.cross_fitted_pseudo_outcome(&sample).unwrap();

        let n = sample.len();
        let folds = estimator.options.folds;
        let mut brute = vec![0.0; n];
        for fold in 0..folds {
            let train: Vec<usize> = (0..n).filter(|i| i % folds != fold).collect();
            let valid: Vec<usize> = (0..n).filter(|i| i % folds == fold).collect();
            let mut gam_ws = GamWorkspace::default();
            let outcome_fit = estimator.fit_outcome(&sample, &train, &mut gam_ws).unwrap();
            let treatment_fit = estimator.fit_treatment(&sample, &train, &mut gam_ws).unwrap();
            let sigma = treatment_sigma(&sample, &train, treatment_fit.as_ref()).unwrap();
            let constant_mean = sample.train_treatment_mean(&train);
            for &i in &valid {
                let mu_observed = predict_one(&outcome_fit, &sample.raw_row(i)).unwrap();
                let treatment_mean = match treatment_fit.as_ref() {
                    Some(fit) => predict_one(fit, &sample.adjustment_row(i)).unwrap(),
                    None => constant_mean,
                };
                let raw_density =
                    gaussian_density(sample.treatment_matrix[i], treatment_mean, sigma);
                let conditional_density = raw_density.max(CONDITIONAL_DENSITY_FLOOR);
                let mut marginal_density = 0.0;
                for &j in &train {
                    let mean_j = match treatment_fit.as_ref() {
                        Some(fit) => predict_one(fit, &sample.adjustment_row(j)).unwrap(),
                        None => constant_mean,
                    };
                    marginal_density += gaussian_density(sample.treatment_matrix[i], mean_j, sigma);
                }
                marginal_density /= train.len() as f64;
                let mut marginal_mu = 0.0;
                for j in 0..n {
                    let mut row = sample.raw_row(j);
                    row[0] = sample.treatment_matrix[i];
                    marginal_mu += predict_one(&outcome_fit, &row).unwrap();
                }
                marginal_mu /= n as f64;
                brute[i] = marginal_mu
                    + (sample.outcome[i] - mu_observed) * marginal_density / conditional_density;
            }
        }
        for (i, (&fast_i, &brute_i)) in fast.values.iter().zip(&brute).enumerate() {
            assert!(
                (fast_i - brute_i).abs() <= 1e-9 * brute_i.abs().max(1.0),
                "row {i}: fast={fast_i} brute={brute_i}"
            );
        }
    }
}

#[test]
fn simultaneous_band_is_never_narrower_than_the_pointwise_band() {
    let (data, a, y, x) = confounded_curve(500);
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: y,
        treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.4, 0.0, 0.4]))),
    });
    let run = |replicates: Option<u32>| {
        let mut estimator = ContinuousResponseEstimator::new([x]);
        estimator.options.bandwidth = Some(0.35);
        estimator.options.simultaneous_replicates = replicates;
        estimator
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap()
    };
    let ResponseUncertainty::PointwiseBand { lower: p_lo, upper: p_hi, .. } = run(None).uncertainty
    else {
        panic!("expected pointwise band");
    };
    let ResponseUncertainty::SimultaneousBand { lower: s_lo, upper: s_hi, .. } =
        run(Some(400)).uncertainty
    else {
        panic!("expected simultaneous band");
    };
    // Both bands standardize by the same influence-based standard error, so the
    // simultaneous band can never be the tighter statement at the same level.
    for index in 0..3 {
        assert!(
            s_lo[index] <= p_lo[index] + 1e-12 && s_hi[index] >= p_hi[index] - 1e-12,
            "simultaneous band narrower than pointwise at {index}"
        );
    }
}

#[test]
fn simultaneous_band_is_deterministic_and_contains_pointwise_curve() {
    let (data, a, y, x) = confounded_curve(500);
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: y,
        treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.4, 0.0, 0.4]))),
    });
    let mut estimator = ContinuousResponseEstimator::new([x]);
    estimator.options.bandwidth = Some(0.35);
    estimator.options.simultaneous_replicates = Some(200);
    let first = estimator
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .unwrap();
    let second = estimator
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .unwrap();
    assert_eq!(first, second);
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &first.estimate
    else {
        panic!("expected surface");
    };
    let ResponseUncertainty::SimultaneousBand { lower, upper, replicates, .. } = &first.uncertainty
    else {
        panic!("expected simultaneous band");
    };
    assert_eq!(*replicates, 200);
    assert!(mean.iter().zip(lower.iter()).zip(upper.iter()).all(|((m, lo), hi)| lo < m && m < hi));
    assert_eq!(first.provenance_id.as_ref(), "estimate.response.kennedy_dr_simultaneous");
}

#[test]
fn identity_shift_has_the_exact_empirical_mean_influence_with_or_without_adjustment() {
    let (data, treatment, outcome, adjustment) = confounded_curve(240);
    for (covariates, interventions) in [
        (vec![adjustment], vec![Intervention::shift(treatment, Value::f64(0.0))]),
        (vec![adjustment], vec![Intervention::shift(treatment, Value::f64(-0.0))]),
        (
            Vec::new(),
            vec![
                Intervention::shift(treatment, Value::f64(0.0)),
                Intervention::shift(adjustment, Value::f64(0.0)),
            ],
        ),
    ] {
        let treatments: Vec<_> =
            interventions.iter().map(|i| i.primary_variable().unwrap()).collect();
        let sample = CompleteSample::read(&data, outcome, &treatments, &covariates).unwrap();
        let mean = sample.outcome.iter().sum::<f64>() / sample.len() as f64;
        let expected: Vec<_> = sample.outcome.iter().map(|value| value - mean).collect();
        let expected_se = (expected.iter().map(|value| value * value).sum::<f64>()
            / (sample.len() * (sample.len() - 1)) as f64)
            .sqrt();
        let (value, uncertainty, _, influence) = ContinuousResponseEstimator::new(covariates)
            .intervention_response(&data, outcome, &interventions)
            .unwrap();
        let ResponseValue::Scalar(value) = value else { panic!("identity is scalar") };
        let ResponseUncertainty::Scalar { standard_error, .. } = uncertainty else {
            panic!("identity has scalar uncertainty")
        };
        assert!((value - mean).abs() < 1e-8);
        assert_eq!(standard_error.to_bits(), expected_se.to_bits());
        assert_eq!(influence.columns[0], expected);
    }
}

#[test]
fn intervention_response_executes_set_shift_and_stochastic_policies() {
    let (data, a, y, x) = confounded_curve(500);
    let estimate = |intervention| {
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: y,
            interventions: Arc::from([intervention]),
        });
        ContinuousResponseEstimator::new([x])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap()
    };
    let set = estimate(Intervention::set(a, Value::f64(0.25)));
    let shift = estimate(Intervention::shift(a, Value::f64(0.25)));
    let stochastic = estimate(Intervention::stochastic(a, StochasticPolicy::gaussian(0.25, 0.01)));
    let scalar = |response: &CausalResponse| match &response.estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => *value,
        _ => panic!("expected scalar"),
    };
    // Linear truth: E[y | do(a=0.25)] = 1 + 2 * 0.25 (E[x] = 0), the same for a shift of 0.25
    // from E[a] ~ 0 and for a tight Gaussian policy centred at 0.25.
    assert!((scalar(&set) - 1.5).abs() < 0.05, "{}", scalar(&set));
    assert!((scalar(&stochastic) - scalar(&set)).abs() < 0.01);
    assert!((scalar(&shift) - 1.5).abs() < 0.05, "{}", scalar(&shift));
    assert_eq!(set.support.status, SupportStatus::Extrapolative);
    assert_eq!(set.provenance_id.as_ref(), "estimate.response.intervention_gcomp");
    let codes: Vec<_> = set.support.warnings.iter().map(|w| w.code.as_ref()).collect();
    assert!(codes.contains(&"response.intervention_plugin_model_dependent"));
    assert!(!codes.contains(&"response.intervention_target_penalized_fallback"));
}

#[test]
fn gaussian_policy_expectation_is_exact_for_a_quadratic_dose_effect() {
    // y = 1 + 2a + a² + 0.8z is a quadratic in the treatment, which the
    // unpenalized cubic treatment smooth reproduces exactly, and z is centred.
    // Under A ~ N(0.25, 0.05²): E[y] = 1 + 2(0.25) + (0.25² + 0.05²) = 1.565 —
    // no sampling error and no fixed policy-integration bias, PROVIDED the
    // treatment and the adjustment covariate are not concurved: `a` is a
    // permutation of the same grid as `z` (decorrelated by a coprime stride),
    // not `0.7*z + small noise` (|corr| < 0.98). With the near-collinear
    // original fixture, the additive GAM cannot uniquely identify the
    // quadratic-in-`a` term from the linear-in-`z` term off the observed
    // (a, z) manifold, and the estimate was biased to ~1.589 regardless of
    // how faithfully the policy integration itself was implemented.
    let n = 500;
    let mut a = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        let z = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
        let treatment = -1.0 + 2.0 * ((i * 173 + 37) % n) as f64 / (n - 1) as f64;
        x.push(z);
        a.push(treatment);
        y.push(1.0 + 2.0 * treatment + treatment * treatment + 0.8 * z);
    }
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("y", y.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap();
    let (a, y, x) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: y,
        interventions: Arc::from([Intervention::stochastic(
            a,
            StochasticPolicy::gaussian(0.25, 0.0025),
        )]),
    });
    let response = ContinuousResponseEstimator::new([x])
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .unwrap();
    let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) = &response.estimate
    else {
        panic!("expected scalar");
    };
    // Finite spline-basis / solver tolerance (GamOptions::tol = 1e-6) leaves a residual
    // ~1.5e-5 even once concurvity is removed; 2e-5 keeps the assertion meaningful
    // (two orders of magnitude tighter than the old 0.024 concurvity bias) without
    // chasing solver-noise digits.
    assert!((value - 1.565).abs() < 2e-5, "value={value}");
}

#[test]
fn intervention_response_discloses_the_penalized_fallback_for_a_binary_treatment() {
    // A binary treatment cannot support the unpenalized cubic target basis;
    // the penalized nuisance fit runs instead, and the result must say so.
    let n = 400;
    let mut a = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        let z = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
        let treatment = if (i * 37 % 101) as f64 / 100.0 < 0.5 + 0.3 * z { 1.0 } else { 0.0 };
        x.push(z);
        a.push(treatment);
        y.push(1.0 + 2.0 * treatment + 0.8 * z + 0.05 * (i as f64).sin());
    }
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("y", y.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap();
    let (a, y, x) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: y,
        interventions: Arc::from([Intervention::set(a, Value::f64(1.0))]),
    });
    let response = ContinuousResponseEstimator::new([x])
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .unwrap();
    let fallback = response
        .support
        .warnings
        .iter()
        .find(|w| w.code.as_ref() == "response.intervention_target_penalized_fallback")
        .expect("the penalized fallback must be disclosed");
    assert!(fallback.fields.iter().any(|(key, _)| key.as_ref() == "target_fit_error"));
    let model = response
        .support
        .warnings
        .iter()
        .find(|w| w.code.as_ref() == "response.intervention_plugin_model_dependent")
        .unwrap();
    assert!(model.message.contains("penalized treatment and adjustment splines"));
    assert!(!model.message.contains("unpenalized treatment splines (the target)"));
}

#[test]
fn intervention_response_fails_closed_for_soft_interventions() {
    let (data, a, y, x) = confounded_curve(100);
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: y,
        interventions: Arc::from([Intervention::soft(
            a,
            antecedent_core::MechanismOverride::named("replacement", Arc::from([])),
        )]),
    });
    let error = ContinuousResponseEstimator::new([x])
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("structural model"));
}

#[test]
fn joint_discrete_mixture_beyond_the_budget_is_refused() {
    // The mixture is a cartesian product, so cost is exponential in the number of joined
    // discrete policies. Four 14-level categoricals is already ~38k combinations per row
    // — seconds of work — and nothing stops a caller going further. Refuse instead of
    // running for hours or silently falling back to Monte Carlo.
    let (data, a, y, x) = confounded_curve(200);
    let run = |levels: usize| {
        let probs: Arc<[f64]> = Arc::from(vec![1.0 / levels as f64; levels]);
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: y,
            interventions: Arc::from([Intervention::stochastic(
                a,
                StochasticPolicy::Categorical { probs },
            )]),
        });
        ContinuousResponseEstimator::new([x]).estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
    };
    assert!(run(MAX_EXACT_MIXTURE_COMBINATIONS).is_ok());
    let error = run(MAX_EXACT_MIXTURE_COMBINATIONS + 1).unwrap_err();
    assert!(error.to_string().contains("exact-mixture budget"), "got {error}");
}

#[test]
fn categorical_intervention_is_an_exact_finite_mixture_not_monte_carlo() {
    // Under an additive model, a Categorical policy on codes {0,1} with weights
    // (0.25, 0.75) must equal 0.25·μ(0) + 0.75·μ(1). Monte Carlo through a continuous
    // spline only approximates that sum and would treat the codes as ordered coordinates.
    let (data, a, y, x) = confounded_curve(500);
    let estimate = |intervention| {
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: y,
            interventions: Arc::from([intervention]),
        });
        let response = ContinuousResponseEstimator::new([x])
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        match response.estimate {
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => value,
            _ => panic!("expected scalar"),
        }
    };
    let at0 = estimate(Intervention::set(a, Value::f64(0.0)));
    let at1 = estimate(Intervention::set(a, Value::f64(1.0)));
    let mixture =
        estimate(Intervention::stochastic(a, StochasticPolicy::categorical([0.25, 0.75])));
    let expected = 0.25 * at0 + 0.75 * at1;
    assert!(
        (mixture - expected).abs() < 1e-10,
        "categorical g-comp={mixture}, exact mixture={expected}"
    );
}

#[test]
fn treatment_sigma_uses_gam_edf_not_n_minus_one() {
    // Discriminating residual-scale check: with edf > 1, dividing by n−1 understates σ.
    let n = 80usize;
    let mut treatments = Vec::with_capacity(n);
    let mut adjustment = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    for i in 0..n {
        let z = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
        adjustment.push(z);
        treatments.push(0.8 * z + 0.1 * (i as f64 * 0.3).sin());
        outcome.push(1.0 + treatments[i] + 0.5 * z);
    }
    let sample = CompleteSample {
        keep: (0..n).collect(),
        outcome,
        treatments: treatments.clone(),
        treatment_matrix: treatments,
        adjustment,
        treatment_cols: 1,
        adjustment_cols: 1,
        raw_cols: 2,
    };
    let train: Vec<usize> = (0..n).collect();
    let mut gam_ws = GamWorkspace::default();
    let fit = ContinuousResponseEstimator::new([VariableId::from_raw(0)])
        .fit_treatment(&sample, &train, &mut gam_ws)
        .unwrap()
        .expect("adjustment present");
    assert!(fit.edf_approx > 1.0 + 1e-6, "edf={}", fit.edf_approx);
    let got = treatment_sigma(&sample, &train, Some(&fit)).unwrap();
    let rss: f64 = fit.residuals.iter().map(|r| r * r).sum();
    let wrong = (rss / (n - 1) as f64).sqrt();
    let right = (rss / (n as f64 - fit.edf_approx).max(1.0)).sqrt();
    assert!(
        (got - right).abs() < 1e-12,
        "treatment_sigma={got} should use edf denominator ({right})"
    );
    assert!(
        (got - wrong).abs() > 1e-6,
        "edf and n-1 denominators coincide; the test cannot discriminate"
    );
    assert!(got > wrong, "edf-aware σ must exceed the n-1 understatement");
}

#[test]
fn row_diagnostics_export_is_opt_in_and_aligned_to_retained_rows() {
    let (data, a, y, x) = confounded_curve(200);
    // Poison one row so a retained-row index gap is observable in the export.
    let mut columns: Vec<Vec<f64>> =
        (0..3).map(|c| data.float64_values(VariableId::from_raw(c)).unwrap()).collect();
    columns[1][7] = f64::NAN;
    let data = TabularData::from_f64_columns([
        ("a", columns[0].as_slice()),
        ("y", columns[1].as_slice()),
        ("x", columns[2].as_slice()),
    ])
    .unwrap();
    let grid = [-0.4, 0.0, 0.4];
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: y,
        treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from(grid))),
    });
    let run = |export: bool| {
        let mut estimator = ContinuousResponseEstimator::new([x]);
        estimator.options.export_row_diagnostics = export;
        estimator
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap()
    };
    let off = run(false);
    assert!(
        off.support.diagnostics.iter().all(|d| !d.id.starts_with("response.row_")),
        "row diagnostics must be absent when the flag is off"
    );
    let on = run(true);
    let diagnostic = |id: &str| {
        on.support
            .diagnostics
            .iter()
            .find(|d| d.id.as_ref() == id)
            .unwrap_or_else(|| panic!("missing diagnostic {id}"))
    };
    let n = 199; // one NaN row dropped
    let row_index = diagnostic("response.row_index");
    assert_eq!(row_index.values.len(), n);
    let expected: Vec<f64> = (0..200).filter(|&i| i != 7).map(f64::from).collect();
    assert_eq!(row_index.values.as_ref(), expected.as_slice());
    let pseudo = diagnostic("response.row_pseudo_outcome");
    assert_eq!(pseudo.values.len(), n);
    assert!(pseudo.values.iter().all(|v| v.is_finite()));
    let influence = diagnostic("response.row_influence");
    assert_eq!(influence.values.len(), grid.len() * n);
    assert!(influence.values.iter().all(|v| v.is_finite()));
    assert!(influence.detail.contains("grid_len=3"));
    // The export must not perturb the estimate itself.
    assert_eq!(off.estimate, on.estimate);
    assert_eq!(off.uncertainty, on.uncertainty);
}

#[test]
fn riesz_ade_calibrates_linear_slope() {
    let (data, a, y, x) = confounded_curve(600);
    let query = ResponseQuery::new(ResponseFunctional::AverageDerivative {
        outcome: y,
        treatment: a,
        weighting: DerivativeWeighting::Observed,
    });
    let response = ContinuousResponseEstimator::new([x])
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .unwrap();
    let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) = response.estimate
    else {
        panic!("expected scalar");
    };
    assert!((value - 2.0).abs() < 0.03, "ade={value}");
    assert_eq!(response.provenance_id.as_ref(), "estimate.response.riesz_ade");
    assert!(
        response.support.diagnostics.iter().any(|d| d.id.as_ref() == "response.outcome_tail_ratio")
    );
}

#[test]
fn bayesian_ade_refits_nuisances_not_frozen_scores() {
    let (data, a, y, x) = confounded_curve(240);
    let estimator = ContinuousResponseEstimator::new([x]);
    let sample = CompleteSample::read(&data, y, &[a], &[x]).unwrap();
    let frozen = estimator.cross_fitted_ade_scores(&sample).unwrap();
    let mut weights = vec![1.0; sample.len()];
    for (i, weight) in weights.iter_mut().enumerate() {
        *weight = if i < sample.len() / 5 { 8.0 } else { 0.2 };
    }
    let folds = estimator.cross_fit_folds(&sample).unwrap();
    let plugin = estimator.weighted_cross_fitted_ade(&sample, &folds, &weights).unwrap();
    let frozen_mean =
        frozen.scores.iter().zip(&weights).map(|(score, weight)| score * weight).sum::<f64>()
            / weights.iter().sum::<f64>();
    assert!(
        (plugin - frozen_mean).abs() > 1e-3,
        "weighted-plugin ADE must move with refitted μ,α; plugin={plugin} frozen={frozen_mean}"
    );

    let query = ResponseQuery::new(ResponseFunctional::AverageDerivative {
        outcome: y,
        treatment: a,
        weighting: DerivativeWeighting::Observed,
    });
    let bayes = crate::BayesianGComputationAte::new().with_n_draws(8);
    let response = estimator
        .estimate_bayesian(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
            &bayes,
            &antecedent_core::ExecutionContext::for_tests(18),
        )
        .unwrap();
    assert_eq!(response.provenance_id.as_ref(), "estimate.response.riesz_ade");
    assert_retained_draws_publish(&response.uncertainty, 8);
    assert!(
        response.assumptions.entries.iter().any(|record| match &record.assumption {
            Assumption::ParametricRestriction(restriction) =>
                restriction.description.contains("not a frozen-score reweight"),
            _ => false,
        }),
        "Bayesian ADE must record the refit assumption: {:?}",
        response.assumptions
    );
    for n_draws in [0, 1] {
        let too_few = crate::BayesianGComputationAte::new().with_n_draws(n_draws);
        let err = estimator
            .estimate_bayesian(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &too_few,
                &antecedent_core::ExecutionContext::for_tests(18),
            )
            .unwrap_err();
        assert!(err.to_string().contains("n_draws >= 2"), "n_draws={n_draws} must refuse: {err}");
    }
}

#[test]
fn point_elasticity_applies_scale_transform() {
    let (data, a, y, x) = confounded_curve(500);
    let query = ResponseQuery::new(ResponseFunctional::PointDerivative {
        outcome: y,
        treatment: a,
        at: 0.4,
        order: 1,
        scale: DerivativeScale::LogLog,
    });
    let mut estimator = ContinuousResponseEstimator::new([x]);
    estimator.options.bandwidth = Some(0.35);
    let response = estimator
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .unwrap();
    let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) = response.estimate
    else {
        panic!("expected scalar");
    };
    // Elasticity a f'(a) / f(a) of f(a) = 1 + 2a (E[x] = 0) at a = 0.4: 0.8 / 1.8.
    assert!((value - 0.8 / 1.8).abs() < 0.03, "elasticity={value}");
    // The elasticity publishes a Fieller interval on the joint
    // local-coordinate covariance and a delta-method standard error.
    let ResponseUncertainty::Scalar { standard_error, lower, upper, .. } = response.uncertainty
    else {
        panic!("expected a published scalar interval, got {:?}", response.uncertainty);
    };
    assert!(standard_error.is_finite() && standard_error > 0.0, "se={standard_error}");
    assert!(lower <= upper, "interval must be ordered");
    assert!(
        response
            .support
            .warnings
            .iter()
            .any(|w| w.code.as_ref() == "response.derivative_interval_fieller"),
        "log-scale elasticity must disclose the Fieller interval"
    );
    assert!(
        !response
            .support
            .warnings
            .iter()
            .any(|w| w.code.as_ref() == "response.derivative_interval_withheld"),
        "log-scale elasticity no longer withholds its interval"
    );
}

#[test]
fn fieller_elasticity_interval_accounts_for_denominator_uncertainty() {
    let covariance = [[0.01, 0.002, 0.0], [0.002, 0.04, 0.0], [0.0; 3]];
    let (lower, upper) =
        fieller_elasticity_interval(1.0, 2.0, 0.5, &covariance, antecedent_stats::NORMAL_Q975)
            .unwrap();
    assert!(lower < 1.0 && 1.0 < upper);
    assert!(upper - 1.0 > 1.0 - lower, "denominator uncertainty makes asymmetric bounds");
    assert!(
        fieller_elasticity_interval(0.1, 2.0, 0.5, &covariance, antecedent_stats::NORMAL_Q975)
            .is_none()
    );
}

#[test]
fn elasticity_refuses_nonpositive_fitted_response() {
    assert!(
        transform_derivative(1.0, 2.0, 0.0, DerivativeScale::LogLog)
            .unwrap_err()
            .to_string()
            .contains("positive fitted response")
    );
    let n: usize = 80;
    let a: Vec<f64> = (0..n).map(|i| -0.4 + i as f64 * 0.01).collect();
    let y: Vec<f64> = a.iter().map(|av| -4.0 - 2.0 * av).collect();
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("y", y.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap();
    let query = ResponseQuery::new(ResponseFunctional::PointDerivative {
        outcome: VariableId::from_raw(1),
        treatment: VariableId::from_raw(0),
        at: 0.2,
        order: 1,
        scale: DerivativeScale::LogLog,
    });
    let mut estimator = ContinuousResponseEstimator::new([VariableId::from_raw(2)]);
    estimator.options.bandwidth = Some(0.35);
    let error = estimator
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("positive fitted response"), "got {error}");
}

#[test]
fn point_derivative_refuses_default_silverman_bandwidth() {
    let (data, a, y, x) = confounded_curve(200);
    let query = ResponseQuery::new(ResponseFunctional::PointDerivative {
        outcome: y,
        treatment: a,
        at: 0.0,
        order: 1,
        scale: DerivativeScale::Identity,
    });
    let error = ContinuousResponseEstimator::new([x])
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("explicit bandwidth"), "got {error}");
}

#[test]
fn static_estimator_refuses_temporal_response_attachment() {
    let (data, a, y, x) = confounded_curve(100);
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: y,
        treatment: ContinuousDomain::new(a, GridSpec::Values(Arc::from([-0.2, 0.2]))),
    })
    .with_temporal(
        antecedent_core::TemporalResponseSpec::new(
            [1u32],
            antecedent_core::TemporalPolicy::pulse(0),
            None,
        )
        .unwrap(),
    );
    let error = ContinuousResponseEstimator::new([x])
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("TemporalResponseEstimator"));
}

#[test]
fn point_derivative_interval_is_robust_bias_corrected() {
    let (data, a, y, x) = confounded_curve(500);
    let at = 0.2;
    let bandwidth = 0.35;
    let query = ResponseQuery::new(ResponseFunctional::PointDerivative {
        outcome: y,
        treatment: a,
        at,
        order: 1,
        scale: DerivativeScale::Identity,
    });
    let mut estimator = ContinuousResponseEstimator::new([x]);
    estimator.options.bandwidth = Some(bandwidth);

    let sample = CompleteSample::read(&data, y, &[a], &[x]).unwrap();
    let pseudo = estimator.cross_fitted_pseudo_outcome(&sample).unwrap().values;
    let local = antecedent_stats::gaussian_local_quadratic_influence(
        &sample.treatments,
        &pseudo,
        at,
        bandwidth,
    )
    .unwrap();
    let corrected = antecedent_stats::gaussian_local_quadratic_bias_corrected(
        &sample.treatments,
        &pseudo,
        at,
        bandwidth,
        None,
    )
    .unwrap();
    let response = estimator
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .unwrap();
    // The point value stays the local-quadratic coordinate.
    let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) = response.estimate
    else {
        panic!("expected a scalar derivative");
    };
    assert!((value - local.point.first_derivative).abs() < 1e-12);
    // The interval is the RBC interval: the bias-corrected coordinate ± z·its
    // own robust SE, not the conventional local-quadratic interval.
    let ResponseUncertainty::Scalar { standard_error, lower, upper, level, interpretation, .. } =
        response.uncertainty
    else {
        panic!("expected scalar uncertainty");
    };
    assert_eq!(interpretation, antecedent_core::IntervalInterpretation::Confidence);
    let z = normal_ppf(0.5 + level / 2.0);
    assert!((standard_error - corrected.robust_first_derivative_standard_error).abs() < 1e-12);
    assert!((0.5 * (lower + upper) - corrected.first_derivative).abs() < 1e-10);
    assert!((upper - lower - 2.0 * z * standard_error).abs() < 1e-10);
    assert!(
        standard_error > local.robust_first_derivative_standard_error,
        "the bias-corrected SE pays for the bias estimate's variance"
    );
    assert!(
        response
            .support
            .warnings
            .iter()
            .any(|w| w.code.as_ref() == "response.derivative_interval_bias_corrected"),
        "a published derivative interval must say it is bias-corrected"
    );
}

#[test]
fn second_derivative_interval_is_centered_at_the_local_quartic_curvature() {
    let (data, a, y, x) = confounded_curve(500);
    let (at, bandwidth) = (0.2, 0.6);
    let query = ResponseQuery::new(ResponseFunctional::PointDerivative {
        outcome: y,
        treatment: a,
        at,
        order: 2,
        scale: DerivativeScale::Identity,
    });
    let mut estimator = ContinuousResponseEstimator::new([x]);
    estimator.options.bandwidth = Some(bandwidth);
    let sample = CompleteSample::read(&data, y, &[a], &[x]).unwrap();
    let pseudo = estimator.cross_fitted_pseudo_outcome(&sample).unwrap().values;
    let corrected = antecedent_stats::gaussian_local_quadratic_bias_corrected(
        &sample.treatments,
        &pseudo,
        at,
        bandwidth,
        None,
    )
    .unwrap();
    let response = estimator
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .unwrap();
    let ResponseUncertainty::Scalar { standard_error, lower, upper, .. } = response.uncertainty
    else {
        panic!("expected scalar uncertainty");
    };
    assert!((standard_error - corrected.robust_second_derivative_standard_error).abs() < 1e-12);
    assert!((0.5 * (lower + upper) - corrected.second_derivative).abs() < 1e-10);
}

#[test]
fn second_derivative_uses_chain_rule_on_log_log_scale() {
    // m(a)=a² at a=2: d² log(m)/d(log(a))² = 0.
    let value = transform_point_derivative(4.0, 4.0, 2.0, 2.0, 2, DerivativeScale::LogLog).unwrap();
    assert!(value.abs() < 1e-12);
}

#[test]
fn plugin_jacobian_recovers_low_dimensional_slopes() {
    let n = 400;
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    let mut y1 = Vec::with_capacity(n);
    let mut y2 = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        let z = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
        let av = z + 0.2 * (i as f64 * 0.7).sin();
        let bv = -0.4 * z + 0.3 * (i as f64 * 1.1).cos();
        a.push(av);
        b.push(bv);
        x.push(z);
        y1.push(1.0 + 2.0 * av - 0.5 * bv + z);
        y2.push(-1.0 + 0.25 * av + 1.5 * bv - 0.7 * z);
    }
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("y1", y1.as_slice()),
        ("y2", y2.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap();
    let query = ResponseQuery::new(ResponseFunctional::Jacobian {
        outcomes: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
        treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        at: Arc::from([0.0, 0.0]),
        scale: DerivativeScale::Identity,
    });
    let response = ContinuousResponseEstimator::new([VariableId::from_raw(4)])
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::IdentifiedUnderParametricRestrictions,
            AssumptionSet::new(),
        )
        .unwrap();
    let ResponseIdentification::PointIdentified(ResponseValue::Jacobian { values, .. }) =
        response.estimate
    else {
        panic!("expected Jacobian");
    };
    for ((got, expected), tolerance) in
        values.iter().zip([2.0, -0.5, 0.25, 1.5]).zip([0.15, 0.03, 0.12, 0.03])
    {
        assert!((got - expected).abs() < tolerance, "got={got}, expected={expected}");
    }
    assert_eq!(response.support.status, SupportStatus::Extrapolative);
    assert_eq!(response.provenance_id.as_ref(), "estimate.response.gam_derivative");
}

#[test]
fn plugin_jacobian_refuses_mismatched_complete_cases() {
    let n: usize = 80;
    let a: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let b: Vec<f64> = (0..n).map(|i| 1.0 - i as f64 / n as f64).collect();
    let y1: Vec<f64> = a.iter().zip(&b).map(|(av, bv)| 1.0 + 2.0 * av - 0.5 * bv).collect();
    let mut y2 = y1.iter().map(|v| -1.0 + 0.25 * v).collect::<Vec<_>>();
    for value in y2.iter_mut().skip(n - 25) {
        *value = f64::NAN;
    }
    let x: Vec<f64> = a.clone();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("y1", y1.as_slice()),
        ("y2", y2.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap();
    let query = ResponseQuery::new(ResponseFunctional::Jacobian {
        outcomes: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
        treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        at: Arc::from([0.5, 0.5]),
        scale: DerivativeScale::Identity,
    });
    let error = ContinuousResponseEstimator::new([VariableId::from_raw(4)])
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::IdentifiedUnderParametricRestrictions,
            AssumptionSet::new(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("shared complete-case"), "got {error}");
}

#[test]
fn plugin_jacobian_warns_when_clamped_outside_support() {
    let n: usize = 120;
    let a: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let b: Vec<f64> = a.clone();
    let y1: Vec<f64> = a.iter().map(|av| 1.0 + 2.0 * av).collect();
    let y2: Vec<f64> = a.iter().map(|av| -1.0 + 0.25 * av).collect();
    let x: Vec<f64> = a.clone();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("y1", y1.as_slice()),
        ("y2", y2.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap();
    let query = ResponseQuery::new(ResponseFunctional::Jacobian {
        outcomes: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
        treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        at: Arc::from([8.0, 8.0]),
        scale: DerivativeScale::Identity,
    });
    let response = ContinuousResponseEstimator::new([VariableId::from_raw(4)])
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::IdentifiedUnderParametricRestrictions,
            AssumptionSet::new(),
        )
        .unwrap();
    assert_eq!(response.support.status, SupportStatus::OutsideEmpiricalSupport);
    assert!(
        response
            .support
            .warnings
            .iter()
            .any(|w| w.code.as_ref() == "response.clamped_basis_derivative"),
        "outside-support Jacobian must emit the clamped-basis warning"
    );
}

/// The plug-in derivatives fit one additive smooth per treatment, so a query over more than
/// two treatments returns one partial per coordinate.
#[test]
fn plugin_derivatives_serve_more_than_two_treatments() {
    let n: usize = 60;
    let col = |k: f64| (0..n).map(|i| (i as f64 * k).sin()).collect::<Vec<f64>>();
    let (a, b, c, x) = (col(0.3), col(0.7), col(1.1), col(0.9));
    // Additive linear truth: the three partials are 2, -1 and 0.5.
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * a[i] - b[i] + 0.5 * c[i]).collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("c", c.as_slice()),
        ("y", y.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap();
    let treatments: Arc<[VariableId]> =
        Arc::from([VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)]);
    let estimator = ContinuousResponseEstimator::new([VariableId::from_raw(4)]);
    let serve = |functional| {
        estimator
            .estimate_identified(
                &data,
                &ResponseQuery::new(functional),
                IdentificationStatus::IdentifiedUnderParametricRestrictions,
                AssumptionSet::new(),
            )
            .expect("a plug-in derivative over three treatments is estimable")
    };
    let jacobian = serve(ResponseFunctional::Jacobian {
        outcomes: Arc::from([VariableId::from_raw(3)]),
        treatments: treatments.clone(),
        at: Arc::from([0.0, 0.0, 0.0]),
        scale: DerivativeScale::Identity,
    });
    let ResponseIdentification::PointIdentified(ResponseValue::Jacobian {
        outcomes,
        treatments: width,
        values: partials,
    }) = jacobian.estimate
    else {
        panic!("expected a point-identified Jacobian");
    };
    assert_eq!((outcomes, width), (1, 3), "one outcome row with one partial per treatment");
    assert_eq!(partials.len(), 3);
    for (got, truth) in partials.iter().zip([2.0, -1.0, 0.5]) {
        assert!((got - truth).abs() < 0.01, "partials {partials:?} vs [2, -1, 0.5]");
    }
    let directional = serve(ResponseFunctional::DirectionalDerivative {
        outcomes: Arc::from([VariableId::from_raw(3)]),
        treatments,
        at: Arc::from([0.0, 0.0, 0.0]),
        direction: Arc::from([1.0, 0.0, 0.0]),
    });
    // Along the first coordinate axis the directional derivative is the first partial of
    // the same additive fit.
    let ResponseIdentification::PointIdentified(value) = directional.estimate else {
        panic!("expected a point-identified directional derivative");
    };
    let along_first = match value {
        ResponseValue::Scalar(value) => value,
        ResponseValue::Vector(values) | ResponseValue::Jacobian { values, .. } => values[0],
        other => panic!("unexpected directional value {other:?}"),
    };
    assert!(
        (along_first - partials[0]).abs() < 1e-9,
        "directional {along_first} vs first partial {}",
        partials[0]
    );
}

/// Every additive-GAM non-convergence refusal keeps its message and names
/// the options that let the backfit settle.
#[test]
fn unfinished_gam_fit_refusals_name_a_remedy() {
    let n: usize = 80;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| 1.0 + 2.0 * v).collect();
    let mut workspace = GamWorkspace::default();
    let mut fit = fit_additive(&x, n, 1, &y, 6, 1.0, &mut workspace).unwrap();
    assert!(fit.converged);
    fit.converged = false;
    for message in [
        GAM_TARGET_NOT_CONVERGED,
        crossfit::GAM_NUISANCE_NOT_CONVERGED,
        crossfit::WEIGHTED_GAM_NUISANCE_NOT_CONVERGED,
    ] {
        let error = require_converged_gam(fit.clone(), message).unwrap_err();
        assert_eq!(error.to_string(), message);
        assert!(!matches!(error, EstimationError::Refused { .. }));
        let remedy = error.remedy().expect("a GAM non-convergence refusal names a remedy");
        assert!(remedy.contains("nuisance_lambda"), "{remedy}");
        assert!(remedy.contains("nuisance_basis"), "{remedy}");
    }
    fit.converged = true;
    assert!(require_converged_gam(fit, GAM_TARGET_NOT_CONVERGED).is_ok());
}

/// Two treatments, two outcomes, one adjustment covariate, Gaussian-like noise.
/// `y1 = scale·(1 + 2a − 0.5b + 0.6a² + x + e₁) + extra·a`.
fn plugin_gradient_data(scale: f64, extra: f64) -> TabularData {
    let n: usize = 400;
    let mut rng = crate::splitmix::FixtureRng::new(0x9E37_79B9_7F4A_7C15);
    let mut noise = || {
        // Sum of four uniforms, centred: a deterministic, roughly Gaussian draw.
        (0..4).map(|_| rng.uniform() - 0.5).sum::<f64>()
    };
    let (mut a, mut b, mut x, mut y1, mut y2) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let z = noise();
        let av = 0.6 * z + noise();
        let bv = -0.3 * z + noise();
        a.push(av);
        b.push(bv);
        x.push(z);
        y1.push(scale * (1.0 + 2.0 * av - 0.5 * bv + 0.6 * av * av + z + noise()) + extra * av);
        y2.push(-1.0 + 0.25 * av + 1.5 * bv - 0.7 * z + noise());
    }
    TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("y1", y1.as_slice()),
        ("y2", y2.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap()
}

fn plugin_jacobian(scale: DerivativeScale) -> ResponseFunctional {
    ResponseFunctional::Jacobian {
        outcomes: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
        treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        at: Arc::from([0.2, -0.1]),
        scale,
    }
}

fn plugin_directional(direction: [f64; 2]) -> ResponseFunctional {
    ResponseFunctional::DirectionalDerivative {
        outcomes: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
        treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        at: Arc::from([0.2, -0.1]),
        direction: Arc::from(direction),
    }
}

fn published(data: &TabularData, functional: ResponseFunctional) -> CausalResponse {
    ContinuousResponseEstimator::new([VariableId::from_raw(4)])
        .estimate_identified(
            data,
            &ResponseQuery::new(functional),
            IdentificationStatus::IdentifiedUnderParametricRestrictions,
            AssumptionSet::new(),
        )
        .unwrap()
}

#[cfg(feature = "calibration-internal")]
fn internal_band(
    data: &TabularData,
    functional: &ResponseFunctional,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (value, uncertainty) = ContinuousResponseEstimator::new([VariableId::from_raw(4)])
        .plugin_gradient_interval_internal(data, functional)
        .unwrap();
    let values = match value {
        ResponseValue::Jacobian { values, .. } | ResponseValue::Vector(values) => values.to_vec(),
        other => panic!("unexpected value {other:?}"),
    };
    let ResponseUncertainty::PointwiseBand { level, lower, upper, interpretation, draws } =
        uncertainty
    else {
        panic!("expected a pointwise band, got {uncertainty:?}");
    };
    assert!((level - ContinuousResponseOptions::default().confidence_level).abs() < 1e-15);
    assert_eq!(interpretation, antecedent_core::IntervalInterpretation::Confidence);
    assert!(draws.is_none());
    (values, lower.to_vec(), upper.to_vec())
}

/// The Frequentist Jacobian and directional derivative publish the point only
/// and say why: the interval route is closed until its coverage is measured.
#[test]
fn frequentist_plugin_gradient_withholds_its_interval_and_says_why() {
    let data = plugin_gradient_data(1.0, 0.0);
    for functional in [plugin_jacobian(DerivativeScale::Identity), plugin_directional([1.0, 1.0])] {
        let response = published(&data, functional);
        assert!(matches!(response.uncertainty, ResponseUncertainty::None));
        let withheld = response
            .support
            .warnings
            .iter()
            .find(|w| w.code.as_ref() == "response.derivative_interval_withheld")
            .expect("the withheld interval is disclosed");
        assert!(withheld.message.contains("not yet measured"), "{}", withheld.message);
        assert!(withheld.message.contains("Bayesian"), "{}", withheld.message);
    }
}

/// The closed route's band is the coefficient sandwich of the published point.
#[cfg(feature = "calibration-internal")]
#[test]
fn plugin_gradient_interval_internal_is_a_sandwich_band_around_the_published_point() {
    let data = plugin_gradient_data(1.0, 0.0);
    let jacobian = plugin_jacobian(DerivativeScale::Identity);
    let (values, lower, upper) = internal_band(&data, &jacobian);
    let ResponseIdentification::PointIdentified(ResponseValue::Jacobian { values: public, .. }) =
        published(&data, jacobian.clone()).estimate
    else {
        panic!("expected Jacobian");
    };
    assert_eq!(values, public.to_vec(), "the band is around the published point");
    for j in 0..4 {
        assert!(lower[j] < values[j] && values[j] < upper[j], "coordinate {j}");
        let half = 0.5 * (upper[j] - lower[j]);
        assert!(half > 0.0 && half < 1.0, "coordinate {j}: half-width {half}");
    }
    // The truth (∂y1/∂a = 2 + 1.2·0.2, ∂y1/∂b = −0.5, ∂y2/∂a = 0.25, ∂y2/∂b = 1.5)
    // sits inside a band of a few half-widths on this one dataset.
    for (j, truth) in [2.24, -0.5, 0.25, 1.5].into_iter().enumerate() {
        let half = 0.5 * (upper[j] - lower[j]);
        assert!((values[j] - truth).abs() < 4.0 * half, "coordinate {j}");
    }

    // A unit direction reads one Jacobian column, band included.
    let (dir, dir_lower, dir_upper) = internal_band(&data, &plugin_directional([1.0, 0.0]));
    for (k, j) in [(0, 0), (1, 2)] {
        assert!((dir[k] - values[j]).abs() < 1e-12);
        assert!((dir_lower[k] - lower[j]).abs() < 1e-9);
        assert!((dir_upper[k] - upper[j]).abs() < 1e-9);
    }
    let ResponseIdentification::PointIdentified(ResponseValue::Vector(public_dir)) =
        published(&data, plugin_directional([1.0, 0.0])).estimate
    else {
        panic!("expected a vector");
    };
    assert_eq!(dir, public_dir.to_vec());

    // Equivariance: scaling y1 scales its point and band; adding an exact
    // linear treatment term shifts the point and leaves the band width alone
    // (the cubic basis spans it, so the residuals do not move).
    let half = |lo: &[f64], hi: &[f64], j: usize| 0.5 * (hi[j] - lo[j]);
    let (scaled, s_lower, s_upper) = internal_band(&plugin_gradient_data(2.0, 0.0), &jacobian);
    let (shifted, t_lower, t_upper) = internal_band(&plugin_gradient_data(1.0, 3.0), &jacobian);
    for j in 0..2 {
        let base = half(&lower, &upper, j);
        assert!((scaled[j] - 2.0 * values[j]).abs() < 1e-6, "coordinate {j}");
        assert!((half(&s_lower, &s_upper, j) - 2.0 * base).abs() < 1e-6 * base.max(1.0));
        let expected_shift = if j == 0 { 3.0 } else { 0.0 };
        assert!((shifted[j] - values[j] - expected_shift).abs() < 1e-6, "coordinate {j}");
        assert!((half(&t_lower, &t_upper, j) - base).abs() < 1e-6 * base.max(1.0));
    }
}

#[cfg(feature = "calibration-internal")]
#[test]
fn plugin_gradient_interval_internal_refuses_what_it_does_not_construct() {
    let data = plugin_gradient_data(1.0, 0.0);
    let estimator = ContinuousResponseEstimator::new([VariableId::from_raw(4)]);
    let error = estimator
        .plugin_gradient_interval_internal(&data, &plugin_jacobian(DerivativeScale::LogLog))
        .unwrap_err();
    assert!(error.to_string().contains("identity scale"), "{error}");
    let error = estimator
        .plugin_gradient_interval_internal(
            &data,
            &ResponseFunctional::PointDerivative {
                outcome: VariableId::from_raw(2),
                treatment: VariableId::from_raw(0),
                at: 0.2,
                order: 1,
                scale: DerivativeScale::Identity,
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("Jacobian and directional"), "{error}");
}

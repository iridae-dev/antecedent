//! Tests for predictive checks, prior sensitivity, MCMC diagnostics, and SBC.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[test]
fn review_prior_sensitivity_verdict_is_unit_invariant() {
    let sensitivity = PriorSensitivity { max_relative_range: 0.25, ..PriorSensitivity::default() };
    for scale in [1e-100, 1.0, 1e100] {
        let summary = PriorSensitivitySummary {
            prior_scales: Arc::from([1.0, 10.0]),
            alphas: Arc::from([]),
            effect_means: Arc::from([scale, 2.0 * scale]),
            effect_sds: Arc::from([scale, scale]),
            ..PriorSensitivitySummary::default()
        };
        let report = sensitivity.to_report(&summary, scale);
        assert!((report.comparison - 0.5).abs() < 1e-12);
        assert!(!report.passed);
    }
}

use super::*;
use std::sync::Arc;

use crate::error::ValidationError;
use antecedent_core::{
    AverageEffectQuery, CausalRng, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{
    BayesianBackendKind, BayesianGCompWorkspace, BayesianGComputationAte, PreparedBayesianProblem,
    SerialDependence,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_identify::IdentificationStatus;
use antecedent_kernels::{quantile_type7_sorted, standard_normal};
use antecedent_prob::{
    ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, HessianFactorization,
    InferenceDiagnostics, PriorSensitivityFamily, PriorSensitivitySummary, PriorSet, PriorSpec,
    sample_inv_gamma,
};
use antecedent_stats::GlmFamily;

#[test]
fn review_predictive_mixture_retains_tail_direction_and_small_variance() {
    let a = summarize_check(PredictiveCheckKind::Posterior, 0.0, &[-2.0; 99], 99);
    let b = summarize_check(PredictiveCheckKind::Posterior, 0.0, &[2.0; 99], 99);
    let mixed = PredictiveCheckReport::mixture_weighted(&[(0.5, &a), (0.5, &b)]).unwrap();
    assert!((mixed.p_value - 1.0).abs() < 1e-12);
    for weight in [1e-200, 1.0, 1e300] {
        let a = report(PredictiveCheckKind::Posterior, 1e12, 1.0);
        let b = report(PredictiveCheckKind::Posterior, 1e12 + 4.0, 1.0);
        let mixed = PredictiveCheckReport::mixture_weighted(&[(weight, &a), (weight, &b)]).unwrap();
        assert!((mixed.predictive_sd - 5.0f64.sqrt()).abs() < 1e-12);
    }
}

fn report(kind: PredictiveCheckKind, mean: f64, sd: f64) -> PredictiveCheckReport {
    PredictiveCheckReport {
        kind,
        observed: mean,
        predictive_mean: mean,
        predictive_sd: sd,
        p_value: 0.4,
        location_tails: [0.2, 0.8],
        observed_dispersion: 1.0,
        predictive_dispersion_mean: 1.0,
        dispersion_p_value: 0.4,
        dispersion_tails: [0.2, 0.8],
        n_sims: 200,
        serial: None,
        residual: Vec::new(),
        residual_p_value: 1.0,
        noise: PredictiveNoise::Likelihood,
    }
}

#[test]
fn mixture_predictive_sd_includes_between_atom_spread() {
    // Two completions that disagree on location but are individually tight.
    // A weighted mean of the SDs would report 1.0 and hide the disagreement.
    let a = report(PredictiveCheckKind::Posterior, 0.0, 1.0);
    let b = report(PredictiveCheckKind::Posterior, 4.0, 1.0);
    let mixed = PredictiveCheckReport::mixture_weighted(&[(0.5, &a), (0.5, &b)]).unwrap();
    assert!((mixed.predictive_mean - 2.0).abs() < 1e-12, "mean={}", mixed.predictive_mean);
    // sqrt(1 + 2^2) = sqrt(5): within-atom variance plus between-atom variance.
    let expected = 5.0f64.sqrt();
    assert!(
        (mixed.predictive_sd - expected).abs() < 1e-12,
        "mixture sd must use total variance, got {} expected {expected}",
        mixed.predictive_sd
    );
    assert_eq!(mixed.n_sims, 200);
}

#[test]
fn mixture_weighted_ignores_zero_weight_and_mismatched_kinds() {
    // A zero-weight leading atom must not decide the mixture's kind.
    let ignored = report(PredictiveCheckKind::Prior, 100.0, 50.0);
    let a = report(PredictiveCheckKind::Posterior, 1.0, 1.0);
    let other_kind = report(PredictiveCheckKind::Prior, 100.0, 50.0);
    let mixed =
        PredictiveCheckReport::mixture_weighted(&[(0.0, &ignored), (1.0, &a), (1.0, &other_kind)])
            .unwrap();
    assert_eq!(mixed.kind, PredictiveCheckKind::Posterior);
    assert!((mixed.predictive_mean - 1.0).abs() < 1e-12, "mean={}", mixed.predictive_mean);
    assert!((mixed.predictive_sd - 1.0).abs() < 1e-12, "sd={}", mixed.predictive_sd);
}

#[test]
fn mixture_weighted_is_none_without_positive_weight() {
    let a = report(PredictiveCheckKind::Posterior, 1.0, 1.0);
    assert!(PredictiveCheckReport::mixture_weighted(&[]).is_none());
    assert!(PredictiveCheckReport::mixture_weighted(&[(0.0, &a)]).is_none());
}

fn toy() -> (TabularData, IdentifiedEstimand, AverageEffectQuery) {
    let n = 60usize;
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "t",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let z: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i] + 0.3 * z[i]).collect();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (TabularData::new(storage), estimand, query)
}

#[test]
fn prior_and_posterior_ppc_run() {
    let (data, estimand, query) = toy();
    let bayes = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: 100,
        seed: 2,
        prior_scale: 10.0,
        ..BayesianGComputationAte::new()
    };
    let prep = bayes.prepare(&data, &estimand, &query).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let prior_rep = PriorPredictiveCheck { n_sims: 50, seed: 3, ..PriorPredictiveCheck::new() }
        .check(&prep, &ctx)
        .unwrap();
    assert_eq!(prior_rep.kind, PredictiveCheckKind::Prior);
    assert!(prior_rep.p_value.is_finite());

    let mut ws = BayesianGCompWorkspace::default();
    let post =
        bayes.fit(&prep, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx).unwrap();
    let post_rep = PosteriorPredictiveCheck { n_sims: 50, ..PosteriorPredictiveCheck::new() }
        .check(&prep, &post)
        .unwrap();
    assert_eq!(post_rep.kind, PredictiveCheckKind::Posterior);
}

#[test]
fn posterior_predictive_replicates_are_the_replicates_the_check_summarizes() {
    use crate::predictive_assessment::ForecastLaw;

    let (data, estimand, query) = toy();
    let bayes = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: 100,
        seed: 2,
        prior_scale: 10.0,
        ..BayesianGComputationAte::new()
    };
    let prep = bayes.prepare(&data, &estimand, &query).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let mut ws = BayesianGCompWorkspace::default();
    let post =
        bayes.fit(&prep, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx).unwrap();
    let check = PosteriorPredictiveCheck { n_sims: 50, ..PosteriorPredictiveCheck::new() };
    let report = check.check(&prep, &post).unwrap();
    let replicates = check.replicates(&prep, &post).unwrap();
    assert_eq!(replicates.draws(), 50);
    assert_eq!(replicates.observations(), prep.design.nrows);
    // Each row's replicates are a sample law; their grand mean is the check's predictive mean.
    let mut total = 0.0;
    let mut count = 0_usize;
    for law in replicates.laws() {
        let ForecastLaw::EmpiricalSample { draws } = law else {
            panic!("a replicate set is scored as sample laws");
        };
        assert_eq!(draws.len(), 50);
        total += draws.iter().sum::<f64>();
        count += draws.len();
    }
    assert!((total / count as f64 - report.predictive_mean).abs() < 1e-9);
}

#[test]
fn summarize_check_observed_outside_range_never_reports_zero() {
    // Simulated draws are tightly clustered; an observation far outside the
    // range on either side must not collapse the Monte Carlo p-value to
    // exactly 0 (D1: below/n or above/n hitting 0 or 1 exactly).
    let n = 200usize;
    let n_sims = u32::try_from(n).expect("test simulation count fits in u32");
    let summaries: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect(); // [0, 1)
    let min_p = 2.0 / (n as f64 + 1.0);

    let low = summarize_check(PredictiveCheckKind::Posterior, -10.0, &summaries, n_sims);
    assert!(low.p_value > 0.0, "p_value must be strictly positive, got {}", low.p_value);
    assert!(low.p_value >= min_p, "p_value {} below the 2/(n+1) floor {min_p}", low.p_value);

    let high = summarize_check(PredictiveCheckKind::Posterior, 10.0, &summaries, n_sims);
    assert!(high.p_value > 0.0, "p_value must be strictly positive, got {}", high.p_value);
    assert!(high.p_value >= min_p, "p_value {} below the 2/(n+1) floor {min_p}", high.p_value);
}

#[test]
fn summarize_check_observed_near_centre_gives_high_p_value() {
    // Sanity check that the corrected formula still behaves as expected in
    // the ordinary case: an observation near the middle of the simulated
    // distribution should give a p-value near 1, not just "not exactly 0".
    let n = 200usize;
    let n_sims = u32::try_from(n).expect("test simulation count fits in u32");
    let summaries: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect(); // [0, 1)
    let centre = summarize_check(PredictiveCheckKind::Posterior, 0.5, &summaries, n_sims);
    assert!(centre.p_value > 0.9, "expected p_value near 1, got {}", centre.p_value);
}

#[test]
fn prior_sensitivity_grid() {
    let (data, estimand, query) = toy();
    let bayes = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: 80,
        seed: 4,
        ..BayesianGComputationAte::new()
    };
    let prep = bayes.prepare(&data, &estimand, &query).unwrap();
    let mut ws = BayesianGCompWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    let sens = PriorSensitivity {
        scales: Arc::from(vec![1.0, 10.0, 50.0]),
        ..PriorSensitivity::standard_grid()
    };
    let (summary, posts) = sens
        .evaluate(&bayes, &prep, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx)
        .unwrap();
    assert_eq!(summary.prior_scales.len(), 3);
    assert!(summary.alphas.is_empty());
    assert_eq!(posts.len(), 3);
    let rep = sens.to_report(&summary, posts[0].summaries.mean[posts[0].effect_column().unwrap()]);
    assert!(rep.passed);
}

#[test]
fn prior_sensitivity_external_alpha_pulls_toward_source() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../conformance/validate/bayesian_checks/expected.json"
    ))
    .unwrap();
    assert!(
        fixture["contracts"]["prior_sensitivity_full_trust_moves_toward_source"].as_bool().unwrap()
    );

    let (data, estimand, query) = toy();
    // Data ATE ≈ 2; bank a tight prior with treatment coef mean = 8.
    let bayes = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: 120,
        seed: 7,
        ..BayesianGComputationAte::new()
    };
    let prep = bayes.prepare(&data, &estimand, &query).unwrap();
    let n = prep.design.ncols;
    let t_col = prep.design.treatment_column().expect("treatment column");
    let mut mean = vec![0.0; n];
    mean[t_col] = 8.0;
    let mut source_prior = PriorSet::new();
    source_prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(mean),
        variance: Arc::from(vec![0.05; n]),
    }));
    let sources = [ExternalPriorSource {
        id: Arc::from("survey_a"),
        prior: source_prior,
        weight: ExternalPriorWeight::power(1.0).unwrap(),
        ess: None,
    }];
    let alphas_applied = [1.0_f64];
    let mut ws = BayesianGCompWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    let sens = PriorSensitivity::standard_alpha_grid();
    let (summary, _) = sens
        .evaluate_external_alpha(
            &bayes,
            &prep,
            IdentificationStatus::NonparametricallyIdentified,
            &mut ws,
            &ctx,
            ExternalAlphaSensitivity { sources: &sources, alphas_applied: &alphas_applied },
        )
        .unwrap();
    assert_eq!(summary.alphas.len(), 5);
    assert!(summary.prior_scales.is_empty());
    assert!(summary.effect_means.iter().all(|m| m.is_finite()));
    let m0 = summary.effect_means[0];
    let m1 = *summary.effect_means.last().unwrap();
    // Full trust (m=1) should sit closer to the banked treatment mean than baseline (m=0).
    assert!(
        (m1 - 8.0).abs() < (m0 - 8.0).abs(),
        "m=1 mean {m1} should be closer to 8 than m=0 mean {m0}"
    );
    let rep = sens.to_report(&summary, m1);
    assert_eq!(rep.refuter.as_ref(), "prior_sensitivity_alpha");
    assert!(rep.informative);
    assert!(rep.comparison.is_finite() && rep.comparison > 0.0);
}

#[test]
fn ppc_catches_variance_misspecification_mean_ok() {
    // Defect C regression: construct data whose outcome mean is exactly zero
    // but has large residual spread (SD ~5), so a model whose replicates have
    // a small cross-observation SD (tight coefficients, residual variance 1)
    // gets the *mean* right (mean-only PPC would pass) while badly
    // understating the *dispersion* of the outcome. The dispersion axis must
    // catch this even though the location axis does not.
    let n = 400usize;
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "t",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    // t, z are both (near-)constant so the fitted/prior mean function has almost
    // no cross-observation spread; y has a large, fixed residual (+/- 5) on top of
    // a tiny signal, so E[y] is well predicted but SD(y) is not.
    let mut rng = CausalRng::from_seed(99);
    let t: Vec<f64> = (0..n).map(|_| 0.0).collect();
    let z: Vec<f64> = (0..n).map(|_| 0.0).collect();
    let mut y: Vec<f64> = (0..n).map(|_| 5.0 * standard_normal(&mut rng)).collect();
    let centre = y.iter().sum::<f64>() / n as f64;
    y.iter_mut().for_each(|v| *v -= centre);
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let data = TabularData::new(storage);

    let bayes = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: 300,
        seed: 5,
        // Tight prior around 0: the mean function is confidently ~0 everywhere,
        // matching E[y]~0, but says nothing spreads out — the mean axis passes.
        prior_scale: 0.05,
        ..BayesianGComputationAte::new()
    };
    let prep = bayes.prepare(&data, &estimand, &query).unwrap();
    let ctx = ExecutionContext::for_tests(1);

    // The prior in force plus a residual variance of 1: replicates spread by ~1.
    let mut prior = bayes.prior_in_force(prep.design.ncols);
    prior.push(PriorSpec::KnownResidualVariance(1.0));
    let rep = PriorPredictiveCheck { n_sims: 300, seed: 6, ..PriorPredictiveCheck::new() }
        .check_with_prior(&prep, &prior, &ctx)
        .unwrap();
    assert_eq!(rep.noise, PredictiveNoise::Likelihood);

    assert!(
        rep.p_value >= 0.05,
        "expected the mean axis to look fine (unbiased predictive mean), got p={}",
        rep.p_value
    );
    assert!(
        rep.dispersion_p_value < 0.05,
        "expected the dispersion axis to catch the 5x variance mismatch, got \
             dispersion_p_value={}",
        rep.dispersion_p_value
    );
    let refuted = rep.to_refutation_report(0.0, 0.05);
    assert!(
        !refuted.passed,
        "predictive check must fail overall when dispersion is badly wrong even \
             though the mean matches"
    );
}

#[test]
fn sbc_to_report_gates_on_uniformity_not_just_mean() {
    // Defect B regression: a U-shaped rank histogram (ranks piled at the two
    // extremes, overdispersed posterior) is symmetric about the middle bin so its
    // mean rank fraction lands squarely in [0.35, 0.65] — a mean-only gate would
    // pass it. The χ² uniformity statistic must catch it.
    let n_reps = 200u32;
    let n_draws = 100usize;
    let n_draws_u32 = u32::try_from(n_draws).expect("test draw count fits in u32");
    // Every replicate's rank is pinned at one of the two extremes (bins 0 and 9),
    // alternating — a textbook U shape.
    let ranks: Vec<u32> = (0..n_reps).map(|i| if i % 2 == 0 { 0 } else { n_draws_u32 }).collect();
    let n_d = n_draws as f64;
    let fracs: Vec<f64> = ranks.iter().map(|&r| f64::from(r) / n_d).collect();
    let mean_rank_frac = fracs.iter().sum::<f64>() / fracs.len() as f64;
    assert!(
        (0.35..=0.65).contains(&mean_rank_frac),
        "fixture sanity: mean rank frac {mean_rank_frac} should look fine on its own"
    );
    let bins = 10usize;
    let bins_u64 = u64::try_from(bins).expect("test bin count fits in u64");
    let n_draws_u64 = u64::try_from(n_draws).expect("test draw count fits in u64");
    let mut counts = vec![0.0; bins];
    for &r in &ranks {
        let b = usize::try_from(((u64::from(r) * bins_u64) / n_draws_u64).min(bins_u64 - 1))
            .expect("test bin index is bounded by the bin count");
        counts[b] += 1.0;
    }
    let expected = f64::from(n_reps) / bins as f64;
    let mut chi2 = 0.0;
    for c in counts {
        let d = c - expected;
        chi2 += d * d / expected.max(1.0);
    }
    assert!(
        chi2 > SBC_CHI2_CRITICAL_9DF_P99,
        "fixture sanity: U-shaped ranks should trip the χ² statistic, got {chi2}"
    );
    let report = SbcReport {
        ranks: Arc::from(ranks),
        mean_rank_frac,
        uniformity_stat: chi2,
        rank_draw_count: n_draws,
        rank_bins: SBC_BINS,
    };
    let sbc = SimulationBasedCalibration { n_reps, n_draws, seed: 0 };
    let rep = sbc.to_report(&report, 2.0);
    assert!(
        !rep.passed,
        "SBC must fail on a U-shaped (non-uniform) rank distribution even though \
             mean_rank_frac={mean_rank_frac:.3} is in [0.35, 0.65]"
    );
}

#[test]
fn sbc_hmc_rank_uses_one_draw_per_independent_chain() {
    let mut diagnostics = InferenceDiagnostics::analytic("hmc");
    diagnostics.n_chains = Some(4);
    diagnostics.ess_bulk_min = Some(5.0);
    let states = [
        0.0, 0.0, 0.0, 1.0, 10.0, 10.0, 10.0, 11.0, 20.0, 20.0, 20.0, 21.0, 30.0, 30.0, 30.0, 31.0,
    ];
    assert_eq!(sbc_rank_draws(&states, &diagnostics).unwrap(), [1.0, 11.0, 21.0, 31.0]);
    diagnostics.n_chains = None;
    assert!(sbc_rank_draws(&states, &diagnostics).is_err());
}

/// Calibration-gate tests for [`SimulationBasedCalibration`]. The slow gate is
/// `#[ignore]`d for the same reason as
/// `antecedent-estimate/src/calibration_coverage.rs`: many refits per test, too
/// slow for every `cargo test`. Run via `cargo test -p antecedent-validate
/// <name> -- --ignored --nocapture`.
mod calibration_gate {
    use super::*;

    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn sbc_conjugate_gaussian_ranks_are_uniform() {
        use antecedent_prob::InvGammaPrior;
        let (data, estimand, query) = toy();
        // SBC needs a residual-variance prior with a finite mean to simulate from
        // (an inverse-gamma shape > 1, or a known variance): the weakly informative
        // default (InvGamma(1e-3, 1e-3)) has neither, so this test supplies one
        // explicitly instead of the isotropic prior_scale shorthand.
        let mut prior = PriorSet {
            specs: vec![PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(
                3, 5.0,
            ))],
            contrast: None,
            categorical: Vec::new(),
            restrictions: Vec::new(),
        };
        prior.push(PriorSpec::ResidualInvGamma(InvGammaPrior { shape: 3.0, scale: 2.0 }));
        let bayes = BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 300,
            seed: 11,
            prior: Some(prior),
            ..BayesianGComputationAte::new()
        };
        let prep = bayes.prepare(&data, &estimand, &query).unwrap();
        let mut ws = BayesianGCompWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        // n_draws + 1 must be a multiple of the 10 rank bins (SimulationBasedCalibration::check).
        let sbc = SimulationBasedCalibration { n_reps: 200, n_draws: 299, seed: 42 };
        let report = sbc
            .check(&bayes, &prep, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx)
            .unwrap();
        let rep = sbc.to_report(&report, 2.0);
        assert!(
            rep.passed,
            "SBC should pass for a correctly specified conjugate Gaussian model: \
                 mean_rank_frac={:.3} chi2={:.3}",
            report.mean_rank_frac, report.uniformity_stat
        );
        assert!(
            (0.35..=0.65).contains(&report.mean_rank_frac),
            "mean_rank_frac={:.3} outside [0.35, 0.65]",
            report.mean_rank_frac
        );
        assert!(
            report.uniformity_stat < SBC_CHI2_CRITICAL_9DF_P99,
            "chi2={:.3} exceeds critical value {SBC_CHI2_CRITICAL_9DF_P99:.3}",
            report.uniformity_stat
        );
    }

    #[test]
    fn sbc_glm_families_rank_on_the_response_scale() {
        use antecedent_prob::BayesLikelihood;
        let (data, estimand, query) = toy();
        let ctx = ExecutionContext::for_tests(17);
        for (family, seed) in [
            (BayesLikelihood::BernoulliLogit, 101),
            (BayesLikelihood::BernoulliProbit, 102),
            (BayesLikelihood::PoissonLog, 103),
        ] {
            let prior = PriorSet {
                specs: vec![PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(
                    3, 0.15,
                ))],
                contrast: None,
                categorical: Vec::new(),
                restrictions: Vec::new(),
            };
            let bayes = BayesianGComputationAte::new()
                .with_likelihood(family)
                .with_n_draws(99)
                .with_seed(seed);
            let bayes = BayesianGComputationAte { prior: Some(prior), ..bayes };
            let prep = bayes.prepare(&data, &estimand, &query).unwrap();
            let mut ws = BayesianGCompWorkspace::default();
            let sbc = SimulationBasedCalibration { n_reps: 100, n_draws: 99, seed };
            let report = sbc
                .check(
                    &bayes,
                    &prep,
                    IdentificationStatus::NonparametricallyIdentified,
                    &mut ws,
                    &ctx,
                )
                .unwrap();
            assert_eq!(report.ranks.len(), 100);
            assert!(
                sbc.to_report(&report, 0.0).passed,
                "{family:?}: mean rank={:.3}, chi2={:.3}",
                report.mean_rank_frac,
                report.uniformity_stat
            );
        }
    }
}

/// Null calibration of the predictive checks: on correctly specified models the
/// two-axis check fails at no more than its nominal rate.
mod predictive_null_calibration {
    use super::*;
    use antecedent_estimate::OverlapPolicy;
    use antecedent_prob::{BayesLikelihood, InvGammaPrior};
    use antecedent_stats::CompiledDesign;

    const DATASETS: u64 = 200;
    const ALPHA: f64 = 0.05;

    fn problem(t: &[f64], x: &[f64], y: &[f64]) -> PreparedBayesianProblem {
        PreparedBayesianProblem {
            design: CompiledDesign::linear_adjustment(t, &[(VariableId::from_raw(2), x)], y, &[])
                .unwrap(),
            method: Arc::from("backdoor.adjustment"),
            adjustment_set: Arc::from([VariableId::from_raw(2)]),
            active: 1.0,
            control: 0.0,
            overlap: OverlapPolicy::ExplicitOverride,
            coef_names: None,
            unit_ids: None,
            serial_dependence: SerialDependence::Iid,
        }
    }

    /// `t ~ Bernoulli(0.5)`, `x ~ N(0, 1)` and the linear predictor
    /// `1 + 0.5 t + 0.8 x` (times `scale`).
    fn covariates(n: usize, rng: &mut CausalRng) -> (Vec<f64>, Vec<f64>) {
        let t = (0..n).map(|_| if rng.next_f64() < 0.5 { 1.0 } else { 0.0 }).collect();
        let x = (0..n).map(|_| standard_normal(rng)).collect();
        (t, x)
    }

    /// Failure rate of the posterior predictive check over correctly specified
    /// datasets drawn by `outcome` and fitted by `estimator`.
    fn posterior_failure_rate(
        estimator: &BayesianGComputationAte,
        outcome: impl Fn(f64, f64, &mut CausalRng) -> f64,
    ) -> f64 {
        let mut failures = 0u32;
        for rep in 0..DATASETS {
            let mut rng = CausalRng::from_seed(0xCA11_B000 + rep);
            let (t, x) = covariates(150, &mut rng);
            let y: Vec<f64> = t.iter().zip(&x).map(|(t, x)| outcome(*t, *x, &mut rng)).collect();
            let prep = problem(&t, &x, &y);
            let ctx = ExecutionContext::for_tests(rep);
            let post = estimator
                .fit(
                    &prep,
                    IdentificationStatus::NonparametricallyIdentified,
                    &mut BayesianGCompWorkspace::default(),
                    &ctx,
                )
                .unwrap();
            let report = PosteriorPredictiveCheck {
                n_sims: 100,
                ..PosteriorPredictiveCheck::for_estimator(estimator, &ctx)
            }
            .check(&prep, &post)
            .unwrap();
            assert_eq!(report.noise, PredictiveNoise::Likelihood);
            failures += u32::from(!report.to_refutation_report(0.0, ALPHA).passed);
        }
        let rate = f64::from(failures) / DATASETS as f64;
        eprintln!("posterior predictive null failure rate {rate:.3} over {DATASETS} datasets");
        rate
    }

    #[test]
    fn posterior_predictive_gaussian_null_rate_is_at_most_nominal() {
        let estimator = BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 100,
            seed: 3,
            ..BayesianGComputationAte::new()
        };
        let rate = posterior_failure_rate(&estimator, |t, x, rng| {
            1.0 + 0.5 * t + 0.8 * x + standard_normal(rng)
        });
        assert!(rate <= 2.0 * ALPHA, "correct Gaussian model failed the PPC at rate {rate}");
    }

    #[test]
    fn posterior_predictive_logit_null_rate_is_at_most_nominal() {
        let estimator = BayesianGComputationAte {
            backend: BayesianBackendKind::Laplace,
            n_draws: 100,
            seed: 4,
            ..BayesianGComputationAte::new()
        }
        .with_likelihood(BayesLikelihood::BernoulliLogit);
        assert_eq!(estimator.glm_family(), GlmFamily::BinomialLogit);
        let rate = posterior_failure_rate(&estimator, |t, x, rng| {
            let p = 1.0 / (1.0 + (-(-0.3 + 0.5 * t + 0.8 * x)).exp());
            if rng.next_f64() < p { 1.0 } else { 0.0 }
        });
        assert!(rate <= 2.0 * ALPHA, "correct logit model failed the PPC at rate {rate}");
    }

    /// Data drawn from a proper prior are a correctly specified prior predictive:
    /// each axis's p-value is (discretely) uniform, so each fails at ≈ α and the
    /// two-axis check at no more than ≈ 1 − (1 − α)².
    #[test]
    fn prior_predictive_proper_prior_null_rate_is_nominal() {
        let n = 100;
        let mut prior = PriorSet {
            specs: vec![PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(
                3, 1.0,
            ))],
            contrast: None,
            categorical: Vec::new(),
            restrictions: Vec::new(),
        };
        let residual = InvGammaPrior { shape: 3.0, scale: 2.0 };
        prior.push(PriorSpec::ResidualInvGamma(residual));
        let (mut location, mut dispersion, mut either) = (0u32, 0u32, 0u32);
        for rep in 0..DATASETS {
            let mut rng = CausalRng::from_seed(0x9A10_0000 + rep);
            let (t, x) = covariates(n, &mut rng);
            let beta: Vec<f64> = (0..3).map(|_| standard_normal(&mut rng)).collect();
            let sigma = sample_inv_gamma(residual.shape, residual.scale, &mut rng).sqrt();
            let y: Vec<f64> = t
                .iter()
                .zip(&x)
                .map(|(t, x)| {
                    beta[0] + beta[1] * t + beta[2] * x + sigma * standard_normal(&mut rng)
                })
                .collect();
            let report =
                PriorPredictiveCheck { n_sims: 200, seed: rep, ..PriorPredictiveCheck::new() }
                    .check_with_prior(
                        &problem(&t, &x, &y),
                        &prior,
                        &ExecutionContext::for_tests(rep),
                    )
                    .unwrap();
            assert_eq!(report.noise, PredictiveNoise::Likelihood);
            location += u32::from(report.p_value < ALPHA);
            dispersion += u32::from(report.dispersion_p_value < ALPHA);
            either += u32::from(!report.to_refutation_report(0.0, ALPHA).passed);
        }
        let rate = |count: u32| f64::from(count) / DATASETS as f64;
        eprintln!(
            "prior predictive null failure rates: location {:.3}, dispersion {:.3}, either {:.3}",
            rate(location),
            rate(dispersion),
            rate(either)
        );
        // Nominal α = 0.05 per axis; 3 Monte Carlo SEs at 200 datasets ≈ 0.046.
        assert!(rate(location) <= 0.1, "location axis null rate {}", rate(location));
        assert!(rate(dispersion) <= 0.1, "dispersion axis null rate {}", rate(dispersion));
        assert!(rate(either) <= 0.16, "two-axis null rate {}", rate(either));
    }

    /// Under the weakly informative residual prior (no finite mean) the prior
    /// predictive scores only the direction the unbounded noise cannot reach.
    #[test]
    fn prior_predictive_improper_residual_prior_is_one_sided_on_dispersion() {
        let mut rng = CausalRng::from_seed(17);
        let (t, x) = covariates(120, &mut rng);
        // Observed spread far above the (tight) mean-only replicates.
        let y: Vec<f64> = t.iter().map(|_| 20.0 * standard_normal(&mut rng)).collect();
        let prep = problem(&t, &x, &y);
        let ctx = ExecutionContext::for_tests(1);
        let tight = PriorSet {
            specs: vec![
                PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(3, 0.01)),
                PriorSpec::ResidualInvGamma(InvGammaPrior::weakly_informative()),
            ],
            contrast: None,
            categorical: Vec::new(),
            restrictions: Vec::new(),
        };
        let report = PriorPredictiveCheck::new().check_with_prior(&prep, &tight, &ctx).unwrap();
        assert_eq!(report.noise, PredictiveNoise::MeanOnlyImproperScale);
        assert!((report.dispersion_tails[1] - 1.0).abs() < f64::EPSILON);
        assert!(report.dispersion_p_value > 0.5, "{report:?}");
        // A diffuse coefficient prior spreads even the means wider than the data.
        let calm: Vec<f64> = t.iter().map(|_| 0.01 * standard_normal(&mut rng)).collect();
        let diffuse = PriorPredictiveCheck::new().check(&problem(&t, &x, &calm), &ctx).unwrap();
        assert!(diffuse.dispersion_p_value < ALPHA, "{diffuse:?}");
        let verdict = diffuse.to_refutation_report(0.0, ALPHA);
        assert!(
            verdict.failure_condition.as_deref().is_some_and(|m| m.contains("diffuse")),
            "{verdict:?}"
        );
    }

    /// The serial axis on a tempered posterior says the interval is tempered.
    #[test]
    fn serial_failure_on_a_tempered_posterior_is_reported_as_informative() {
        let mut report = report(PredictiveCheckKind::Posterior, 0.0, 1.0);
        report.serial = Some(SerialDiscrepancy {
            lag: 1,
            observed: 0.6,
            predictive_mean: 0.0,
            p_value: 0.01,
            tails: [0.995, 0.005],
            tempering_kappa: Some(3.2),
        });
        let verdict = report.to_refutation_report(0.0, ALPHA);
        assert!(!verdict.passed);
        let message = verdict.failure_condition.unwrap();
        assert!(message.contains("already tempered") && message.contains("kappa=3.2"), "{message}");
        let untempered = PredictiveCheckReport::mixture_weighted(&[
            (1.0, &report),
            (1.0, &{
                let mut other = report.clone();
                other.serial =
                    other.serial.map(|s| SerialDiscrepancy { tempering_kappa: None, ..s });
                other
            }),
        ])
        .unwrap();
        assert_eq!(untempered.serial.unwrap().tempering_kappa, None);
    }

    /// A linear Gaussian model fitted to `y = 1 + 0.5 t + 0.8 x + 1.5 x² + noise`: the mean
    /// and dispersion axes are matched by construction (the normal equations reproduce the
    /// mean, the replicate SD is drawn to match the residual SD), so only the residual-shape
    /// axes can see the neglected curvature.
    #[test]
    fn posterior_predictive_flags_a_nonlinear_mean() {
        let mut rng = CausalRng::from_seed(0xBEEF);
        let (t, x) = covariates(150, &mut rng);
        let y: Vec<f64> = t
            .iter()
            .zip(&x)
            .map(|(t, x)| 1.0 + 0.5 * t + 0.8 * x + 1.5 * x * x + 0.3 * standard_normal(&mut rng))
            .collect();
        let prep = problem(&t, &x, &y);
        let estimator = BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 200,
            seed: 3,
            ..BayesianGComputationAte::new()
        };
        let ctx = ExecutionContext::for_tests(1);
        let post = estimator
            .fit(
                &prep,
                IdentificationStatus::NonparametricallyIdentified,
                &mut BayesianGCompWorkspace::default(),
                &ctx,
            )
            .unwrap();
        let report = PosteriorPredictiveCheck {
            n_sims: 200,
            ..PosteriorPredictiveCheck::for_estimator(&estimator, &ctx)
        }
        .check(&prep, &post)
        .unwrap();
        assert_eq!(report.residual.len(), 4);
        assert!(report.residual_p_value < ALPHA, "{report:?}");
        let verdict = report.to_refutation_report(0.0, ALPHA);
        assert!(!verdict.passed, "{verdict:?}");
        let message = verdict.failure_condition.unwrap();
        assert!(message.contains("residual-shape"), "{message}");
        // The curvature is what the statistics see: observed residuals correlate positively
        // with the squared fitted-mean deviation, the replicates do not.
        let curvature =
            report.residual.iter().find(|r| r.statistic == ResidualStatistic::Curvature).unwrap();
        assert!(curvature.observed > 0.3, "{curvature:?}");
        assert!(curvature.predictive_mean.abs() < 0.1, "{curvature:?}");
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the asserted values are exact by construction of the fixture (zeros, ones, small dyadic rationals), so exact equality is intended"
    )]
    fn prior_predictive_check_has_no_residual_axes() {
        let mut rng = CausalRng::from_seed(23);
        let (t, x) = covariates(60, &mut rng);
        let y: Vec<f64> = t.iter().map(|_| standard_normal(&mut rng)).collect();
        let report = PriorPredictiveCheck::new()
            .check(&problem(&t, &x, &y), &ExecutionContext::for_tests(1))
            .unwrap();
        assert!(report.residual.is_empty());
        assert_eq!(report.residual_p_value, 1.0);
    }

    #[test]
    fn strided_draw_selection_spreads_over_the_whole_store() {
        // Identity when every draw is used.
        assert_eq!(
            (0..5).map(|k| strided_draw_index(k, 5, 5)).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4]
        );
        // 4 of 10 draws: k * 10 / 4 = 0, 2, 5, 7; 200 of 400 chain-concatenated draws reach
        // the last chain instead of stopping at the 200th draw.
        assert_eq!((0..4).map(|k| strided_draw_index(k, 4, 10)).collect::<Vec<_>>(), [0, 2, 5, 7]);
        assert_eq!(strided_draw_index(199, 200, 400), 398);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the asserted values are exact by construction of the fixture (zeros, ones, small dyadic rationals), so exact equality is intended"
    )]
    fn mixture_averages_residual_tails_and_omnibus_p_over_atoms() {
        let axis = |tails: [f64; 2]| ResidualDiscrepancy {
            statistic: ResidualStatistic::Skewness,
            observed: 1.0,
            predictive_mean: 0.0,
            p_value: (2.0 * tails[0].min(tails[1])).min(1.0),
            tails,
        };
        let mut a = report(PredictiveCheckKind::Posterior, 0.0, 1.0);
        a.residual = vec![axis([0.1, 0.9])];
        a.residual_p_value = 0.2;
        let mut b = report(PredictiveCheckKind::Posterior, 0.0, 1.0);
        b.residual = vec![axis([0.3, 0.7])];
        b.residual_p_value = 0.6;
        let mixed = PredictiveCheckReport::mixture_weighted(&[(1.0, &a), (1.0, &b)]).unwrap();
        assert_eq!(mixed.residual.len(), 1);
        assert!((mixed.residual[0].tails[0] - 0.2).abs() < 1e-12);
        assert!((mixed.residual[0].tails[1] - 0.8).abs() < 1e-12);
        assert!((mixed.residual[0].p_value - 0.4).abs() < 1e-12);
        assert!((mixed.residual_p_value - 0.4).abs() < 1e-12);
        // An atom without the axis drops it from the mixture.
        let bare = report(PredictiveCheckKind::Posterior, 0.0, 1.0);
        let dropped = PredictiveCheckReport::mixture_weighted(&[(1.0, &a), (1.0, &bare)]).unwrap();
        assert!(dropped.residual.is_empty());
        assert_eq!(dropped.residual_p_value, 1.0);
    }

    #[test]
    fn replicate_calibration_checks_stop_on_cancellation() {
        let (data, estimand, query) = toy();
        let mut prior = PriorSet {
            specs: vec![PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(
                3, 5.0,
            ))],
            contrast: None,
            categorical: Vec::new(),
            restrictions: Vec::new(),
        };
        prior.push(PriorSpec::ResidualInvGamma(InvGammaPrior { shape: 3.0, scale: 2.0 }));
        let bayes = BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 99,
            seed: 1,
            prior: Some(prior),
            ..BayesianGComputationAte::new()
        };
        let prep = bayes.prepare(&data, &estimand, &query).unwrap();
        let mut ws = BayesianGCompWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        ctx.cancellation.cancel();
        let sbc = SimulationBasedCalibration { n_reps: 50, n_draws: 99, seed: 0 }.check(
            &bayes,
            &prep,
            IdentificationStatus::NonparametricallyIdentified,
            &mut ws,
            &ctx,
        );
        assert!(matches!(sbc, Err(ValidationError::Cancelled)), "{sbc:?}");
    }

    #[test]
    fn sbc_refuses_configurations_the_rank_theorem_cannot_support() {
        let (data, estimand, query) = toy();
        let bayes = BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 50,
            seed: 1,
            ..BayesianGComputationAte::new()
        };
        let prep = bayes.prepare(&data, &estimand, &query).unwrap();
        let mut ws = BayesianGCompWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let mut run = |sbc: SimulationBasedCalibration| {
            sbc.check(
                &bayes,
                &prep,
                IdentificationStatus::NonparametricallyIdentified,
                &mut ws,
                &ctx,
            )
        };
        // 101 ranks do not divide into 10 equal-mass bins.
        assert!(matches!(
            run(SimulationBasedCalibration { n_reps: 50, n_draws: 100, seed: 0 }),
            Err(ValidationError::NotApplicable { .. })
        ));
        // Fewer than five expected replicates per bin.
        assert!(matches!(
            run(SimulationBasedCalibration { n_reps: 49, n_draws: 99, seed: 0 }),
            Err(ValidationError::NotApplicable { .. })
        ));
        assert!(matches!(
            run(SimulationBasedCalibration::new(1)),
            Err(ValidationError::NotApplicable { .. })
        ));
        // The weakly informative default residual prior (InvGamma(1e-3, 1e-3)) has no finite
        // mean: it cannot generate data, so the check does not pretend to certify it.
        let refused = run(SimulationBasedCalibration { n_reps: 50, n_draws: 99, seed: 0 });
        assert!(
            matches!(refused, Err(ValidationError::NotApplicable { message }) if message.contains("finite mean")),
            "{refused:?}"
        );
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the asserted values are exact by construction of the fixture (zeros, ones, small dyadic rationals), so exact equality is intended"
    )]
    fn sbc_generating_parameters_follow_the_fitted_prior() {
        // beta_c | sigma^2 ~ N(mu_c, sigma^2 V0_c): known sigma^2 = 4, V0 = (0.25, 4), so
        // beta_0 ~ N(2, 1) and beta_1 ~ N(-1, 16).
        let coefficients = GaussianCoefficientPrior {
            mean: Arc::from(vec![2.0, -1.0]),
            variance: Arc::from(vec![0.25, 4.0]),
        };
        let mut rng = CausalRng::from_seed(5);
        let draws = 20_000_usize;
        let (mut b0, mut b1) = (Vec::new(), Vec::new());
        for _ in 0..draws {
            let (sigma2, beta) = draw_generating_parameters(
                &coefficients,
                None,
                GeneratingVariance::Known(4.0),
                &mut rng,
            );
            assert_eq!(sigma2, 4.0);
            b0.push(beta[0]);
            b1.push(beta[1]);
        }
        let stats = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            (m, v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64)
        };
        let ((m0, v0), (m1, v1)) = (stats(&b0), stats(&b1));
        assert!((m0 - 2.0).abs() < 0.05 && (v0 - 1.0).abs() < 0.1, "beta_0: {m0}, {v0}");
        assert!((m1 + 1.0).abs() < 0.25 && (v1 - 16.0).abs() < 1.5, "beta_1: {m1}, {v1}");
        // InvGamma(5, 4) has mean 1: the drawn sigma^2 average to it.
        let sigma2_mean = (0..draws)
            .map(|_| {
                draw_generating_parameters(
                    &coefficients,
                    None,
                    GeneratingVariance::InvGamma { shape: 5.0, scale: 4.0 },
                    &mut rng,
                )
                .0
            })
            .sum::<f64>()
            / draws as f64;
        assert!((sigma2_mean - 1.0).abs() < 0.03, "sigma2 mean {sigma2_mean}");
        // A dense prior (coefficient correlation 0.7) draws correlated coefficients.
        let corr = antecedent_prob::CoefficientCorrelation::new(2, vec![1.0, 0.7, 0.7, 1.0])
            .unwrap()
            .cholesky()
            .unwrap();
        let pairs: Vec<(f64, f64)> = (0..draws)
            .map(|_| {
                let (_, b) = draw_generating_parameters(
                    &coefficients,
                    Some(&corr),
                    GeneratingVariance::Known(4.0),
                    &mut rng,
                );
                (b[0], b[1])
            })
            .collect();
        let (a0, a1): (Vec<f64>, Vec<f64>) = pairs.iter().copied().unzip();
        let ((m0, v0), (m1, v1)) = (stats(&a0), stats(&a1));
        let cov = pairs.iter().map(|(x, y)| (x - m0) * (y - m1)).sum::<f64>() / (draws - 1) as f64;
        let rho = cov / (v0 * v1).sqrt();
        assert!((rho - 0.7).abs() < 0.02, "dense prior draw correlation {rho}");
    }

    #[test]
    fn sbc_bins_have_equal_prior_mass_when_the_ranks_divide_evenly() {
        // 99 draws -> ranks 0..=99 (100 of them): every one of the 10 bins holds exactly 10.
        let mut counts = [0usize; SBC_BINS];
        for rank in 0..=99_u32 {
            counts[sbc_bin(rank, 99)] += 1;
        }
        assert_eq!(counts, [10; SBC_BINS]);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the asserted values are exact by construction of the fixture (zeros, ones, small dyadic rationals), so exact equality is intended"
    )]
    fn quantile_interpolates_linearly_between_order_statistics() {
        let odd = [0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(quantile_type7_sorted(&odd, 0.5), 2.0);
        assert!((quantile_type7_sorted(&odd, 0.9) - 3.6).abs() < 1e-12);
        assert_eq!(quantile_type7_sorted(&odd, 0.0), 0.0);
        assert_eq!(quantile_type7_sorted(&odd, 1.0), 4.0);
        // The median of an even count is the midpoint, not an order statistic.
        assert_eq!(quantile_type7_sorted(&[1.0, 2.0, 3.0, 4.0], 0.5), 2.5);
        assert!(quantile_type7_sorted(&[], 0.5).is_nan());
    }

    #[test]
    fn mcmc_failure_text_names_the_failing_clause() {
        let mut d = InferenceDiagnostics::analytic("hmc");
        assert!(mcmc_report(&d, 1.0).is_none(), "non-MCMC diagnostics are not applicable");
        d.factorization = HessianFactorization::Mcmc;
        d.converged = true;
        d.n_chains = Some(4);
        d.n_warmup = Some(100);
        d.ess_bulk_min = Some(900.0);
        d.ess_tail_min = Some(900.0);
        d.rhat_max = Some(1.004);
        d.n_postwarmup_divergences = Some(0);
        d.mean_accept_prob = Some(0.8);
        d.max_abs_delta_h = Some(0.1);
        d.all_chains_moved = Some(true);
        let healthy = mcmc_report(&d, 1.0).unwrap();
        assert!(healthy.passed && healthy.failure_condition.is_none());
        // Tail ESS and chain movement fail; bulk ESS, R-hat and divergences are fine, so the
        // old text (rhat, bulk ESS, divergences) would have named nothing wrong.
        d.ess_tail_min = Some(50.0);
        d.all_chains_moved = Some(false);
        let sick = mcmc_report(&d, 1.0).unwrap();
        assert!(!sick.passed);
        let message = sick.failure_condition.unwrap();
        assert!(message.contains("tail ESS") && message.contains("chain moved"), "{message}");
        assert!(!message.contains("split-R̂"), "{message}");
    }

    #[test]
    fn mcmc_publication_gate_checks_numeric_boundaries_and_missing_values() {
        // The pinned oracle contract is a looser diagnostics-routing check
        // (R-hat 1.05 / ESS 10); publication uses the library's documented
        // stricter floor (R-hat 1.01 / ESS 100 per chain). Exercise the
        // actual publication boundary numerically so a missing metric cannot
        // silently become a pass.
        let mut d = InferenceDiagnostics::analytic("hmc");
        d.factorization = HessianFactorization::Mcmc;
        d.converged = true;
        d.n_chains = Some(4);
        d.n_warmup = Some(100);
        d.ess_bulk_min = Some(400.0);
        d.ess_tail_min = Some(400.0);
        d.rhat_max = Some(1.01);
        d.n_postwarmup_divergences = Some(0);
        d.mean_accept_prob = Some(0.8);
        d.max_abs_delta_h = Some(0.1);
        d.all_chains_moved = Some(true);
        assert!(mcmc_report(&d, 0.0).unwrap().passed, "inclusive boundary must pass");

        for (mutated, expected_clause) in [
            (
                {
                    let mut x = d.clone();
                    x.rhat_max = Some(1.010_001);
                    x
                },
                "split-R̂",
            ),
            (
                {
                    let mut x = d.clone();
                    x.ess_bulk_min = Some(399.999);
                    x
                },
                "bulk ESS",
            ),
            (
                {
                    let mut x = d.clone();
                    x.ess_tail_min = None;
                    x
                },
                "tail ESS",
            ),
            (
                {
                    let mut x = d.clone();
                    x.n_postwarmup_divergences = None;
                    x
                },
                "divergences",
            ),
        ] {
            let report = mcmc_report(&mutated, 0.0).unwrap();
            assert!(!report.passed, "mutation should fail: {expected_clause}");
            assert!(
                report.failure_condition.unwrap().contains(expected_clause),
                "failure should name {expected_clause}"
            );
        }
    }

    #[test]
    fn pinned_external_diagnostics_thresholds_are_distinct_from_publication() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../conformance/validate/bayesian_checks/expected.json"
        ))
        .unwrap();
        let oracle = &fixture["contracts"];
        let oracle_rhat = oracle["mcmc_max_rhat"].as_f64().unwrap();
        let oracle_ess = oracle["mcmc_min_bulk_ess"].as_f64().unwrap();
        assert!((oracle_rhat - 1.05).abs() < f64::EPSILON);
        assert!((oracle_ess - 10.0).abs() < f64::EPSILON);
        // These oracle bounds define only whether ArviZ-style diagnostic
        // values agree with the pinned external contract. The production
        // publication gate remains 1.01 and 100 effective draws per chain.
        let publication_rhat_limit = 1.01;
        let publication_ess_limit_per_chain = 100.0;
        assert!(oracle_rhat > publication_rhat_limit);
        assert!(oracle_ess < publication_ess_limit_per_chain);
    }

    #[test]
    fn prior_sensitivity_grid_follows_the_prior_in_force() {
        let (data, estimand, query) = toy();
        let plain = BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 40,
            seed: 4,
            ..BayesianGComputationAte::new()
        };
        let prep = plain.prepare(&data, &estimand, &query).unwrap();
        let isotropic = PriorSensitivity::for_estimator(&plain);
        assert!(!isotropic.scales.is_empty() && isotropic.variance_multipliers.is_empty());
        let supplied = BayesianGComputationAte {
            prior: Some(PriorSet::weakly_informative(prep.design.ncols)),
            ..plain.clone()
        };
        let resolved = PriorSensitivity::for_estimator(&supplied);
        assert!(resolved.scales.is_empty() && !resolved.variance_multipliers.is_empty());
        let mut ws = BayesianGCompWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        // Dispatch reaches the resolved-prior family for a supplied prior...
        let (summary, _) = resolved
            .evaluate_in_force(
                &supplied,
                &prep,
                IdentificationStatus::NonparametricallyIdentified,
                &mut ws,
                &ctx,
            )
            .unwrap();
        assert_eq!(summary.family, PriorSensitivityFamily::ResolvedPriorVariance);
        // ...and the isotropic grid refuses to replace it.
        assert!(
            PriorSensitivity::standard_grid()
                .evaluate(
                    &supplied,
                    &prep,
                    IdentificationStatus::NonparametricallyIdentified,
                    &mut ws,
                    &ctx,
                )
                .is_err()
        );
    }

    #[test]
    fn resolved_prior_sensitivity_matches_orthogonal_gaussian_posterior_means() {
        // With balanced t in {-1, 1}, X'X = diag(100, 100) and
        // X'y = (0, 200). A N(0, v) coefficient prior with known
        // residual variance one gives E[beta_t | y] = 200/(100+1/v).
        // The requested treatment contrast is 1 - (-1) = 2.
        let n = 100usize;
        let treatment: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { -1.0 } else { 1.0 }).collect();
        let outcome: Vec<f64> = treatment.iter().map(|t| 2.0 * t).collect();
        let design =
            antecedent_stats::CompiledDesign::linear_adjustment(&treatment, &[], &outcome, &[])
                .unwrap();
        let problem = PreparedBayesianProblem {
            design,
            method: Arc::from("backdoor.adjustment"),
            adjustment_set: Arc::from([]),
            active: 1.0,
            control: -1.0,
            overlap: antecedent_estimate::OverlapPolicy::ExplicitOverride,
            coef_names: None,
            unit_ids: None,
            serial_dependence: SerialDependence::Iid,
        };
        let mut prior = PriorSet::new();
        prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(2, 0.1)));
        prior.push(PriorSpec::KnownResidualVariance(1.0));
        let estimator =
            BayesianGComputationAte::conjugate().with_n_draws(200).with_seed(91).with_prior(prior);
        let grid = PriorSensitivity {
            variance_multipliers: Arc::from([0.25, 1.0, 4.0]),
            scales: Arc::from([]),
            alphas: Arc::from([]),
            max_relative_range: 1.0,
        };
        let (summary, _) = grid
            .evaluate_in_force(
                &estimator,
                &problem,
                IdentificationStatus::NonparametricallyIdentified,
                &mut BayesianGCompWorkspace::default(),
                &ExecutionContext::for_tests(91),
            )
            .unwrap();
        for (&multiplier, &observed) in
            grid.variance_multipliers.iter().zip(summary.effect_means.iter())
        {
            let variance = 0.01 * multiplier;
            let expected = 400.0 / (100.0 + 1.0 / variance);
            assert!((observed - expected).abs() < 0.12, "{multiplier}: {observed} vs {expected}");
        }
    }
}

#[test]
fn bayesian_suite_does_not_drop_a_custom_validator() {
    use antecedent_estimate::BayesianGCompWorkspace;

    use crate::custom::CustomEffectValidator;
    use crate::error::ValidationError;
    use crate::suite::{BayesianSuiteContext, ValidationSuite, ValidatorId};

    struct Kept;
    impl CustomEffectValidator for Kept {
        fn name(&self) -> &'static str {
            "kept"
        }
        fn validate(
            &self,
            _: &crate::RefutationProblem<'_>,
            _: &ExecutionContext,
        ) -> Result<crate::RefutationReport, ValidationError> {
            panic!("the Bayesian path must refuse before calling the custom validator");
        }
    }

    let (data, estimand, query) = toy();
    let bayes = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: 20,
        seed: 2,
        ..BayesianGComputationAte::new()
    };
    let prep = bayes.prepare(&data, &estimand, &query).unwrap();
    let mut ws = BayesianGCompWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    let post =
        bayes.fit(&prep, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx).unwrap();
    let mut bayes_ctx = BayesianSuiteContext::new(
        &bayes,
        &prep,
        &post,
        IdentificationStatus::NonparametricallyIdentified,
        &mut ws,
        0.0,
    );
    let refused = ValidationSuite::new()
        .with(ValidatorId::McmcDiagnostics)
        .with_custom(Arc::new(Kept))
        .run_bayesian(&mut bayes_ctx, &ctx)
        .unwrap_err();
    assert!(
        matches!(
            refused,
            ValidationError::NotApplicable { message }
                if message.contains("refuses to drop")
        ),
        "{refused}"
    );
    let kept = ValidationSuite::new()
        .with(ValidatorId::McmcDiagnostics)
        .run_bayesian(&mut bayes_ctx, &ctx)
        .unwrap();
    assert_eq!(kept.len(), 1, "a suite with no custom validator still runs");
}

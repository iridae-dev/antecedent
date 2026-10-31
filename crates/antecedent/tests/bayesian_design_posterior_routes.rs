//! The unlicensed Bayesian counterparts of the randomized survival and fixed group-dose policy
//! routes, through the public `Study` / `PreparedStudy` flow.
//!
//! Each posterior answers the same target as the frequentist row on the same coordinate and
//! is checked against an independent reference: the survival posterior against the gamma
//! conjugate updates, closed-form survival means and numerical quadrature restricted means of
//! its piecewise-exponential model (computed by hand and by quadrature, not by the
//! implementation) and, under a flat prior, against the frequentist product limit of the same
//! rows; the dose-policy posterior against the per-group ridge posterior of numpy with scipy
//! window quadrature and, on a long fixture, against the frequentist kernel estimate. The
//! retained joint draws, the contracted artifact and its tamper refusals are exercised, and
//! every refusal names its next attempt. None of these routes is licensed.
#![allow(
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    reason = "integration test: exact markers are compared and fixture indices are small"
)]

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{
    BayesianConfig, DesignPosteriorModel, InferenceMode, PrimaryEstimate, Study, StudyResult,
};
use antecedent_core::{
    CausalQuery, ContinuousDoseResponseQuery, FixedGroupDosePolicy, ObservationAssumption,
    SurvivalFunctional, SurvivalQuery, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::dose_basis_regression::ResidualPrior;

fn close(left: f64, right: f64, tolerance: f64) {
    assert!((left - right).abs() <= tolerance, "{left} != {right}");
}

/// The artifact encoder refuses a forged wire with this convert error naming `reason`.
fn assert_refused<T>(outcome: Result<T, impl std::fmt::Display>, reason: &str, what: &str) {
    let text =
        outcome.err().unwrap_or_else(|| panic!("{what}: the forged wire was encoded")).to_string();
    assert!(text.contains("convert error") && text.contains(reason), "{what}: {text}");
}

// ---------------------------------------------------------------------------------------------
// Survival
// ---------------------------------------------------------------------------------------------

fn survival_query(tau: f64, functional: SurvivalFunctional) -> SurvivalQuery {
    SurvivalQuery {
        duration: VariableId::from_raw(0),
        event: VariableId::from_raw(1),
        treatment: VariableId::from_raw(2),
        tau,
        delayed_entry: None,
        known_censoring: None,
        observation_assumption: ObservationAssumption::IndependentGiven(Arc::from([])),
        functional,
    }
}

fn survival_table(duration: &[f64], event: &[f64], treatment: &[f64]) -> TabularData {
    TabularData::from_f64_columns([
        ("duration", duration),
        ("event", event),
        ("treatment", treatment),
    ])
    .unwrap()
}

/// Six control and six treated subjects: events and exposures by hand are
/// control 2, 1, 1 events over 5.4, 3.0, 1.5 at risk time and treated 0, 1, 0 over 5.7, ...
fn small_survival() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut duration = vec![0.5, 1.2, 2.5, 3.5, 1.8, 0.9];
    let mut event = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
    let mut treatment = vec![0.0; 6];
    duration.extend([0.7, 1.5, 2.2, 3.2, 3.8, 1.1]);
    event.extend([0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
    treatment.extend([1.0; 6]);
    (duration, event, treatment)
}

fn bayesian(draws: usize, model: DesignPosteriorModel) -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(draws).design_model(model))
}

fn hazards(cuts: &[f64], shape: f64, rate: f64) -> DesignPosteriorModel {
    DesignPosteriorModel::PiecewiseHazard {
        cuts: Arc::from(cuts),
        prior_shape: shape,
        prior_rate: rate,
    }
}

fn run(
    data: &TabularData,
    query: CausalQuery,
    inference: InferenceMode,
    seed: u64,
) -> Result<(StudyResult, antecedent::PreparedStudy, ExecutionContext), antecedent::CausalError> {
    let ctx = ExecutionContext::for_tests(seed);
    let study = Study::tabular(data.clone()).query(query).inference(inference).build()?;
    let prepared = study.prepare(&ctx)?;
    let result = prepared.estimate(data, &ctx)?;
    Ok((result, prepared, ctx))
}

#[test]
fn survival_posterior_matches_the_gamma_hazard_reference_and_keeps_joint_draws() {
    let (duration, event, treatment) = small_survival();
    let data = survival_table(&duration, &event, &treatment);
    let query = CausalQuery::Survival(survival_query(3.0, SurvivalFunctional::SurvivalAndRmst));
    let (result, _, _) =
        run(&data, query, bayesian(40_000, hazards(&[1.0, 2.0, 3.0], 1.0, 1.0)), 11).unwrap();

    assert!(matches!(result.estimate(), PrimaryEstimate::NotAnEffect));
    assert!(result.survival().is_none());
    assert_eq!(result.support_status(), None);
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("bayesian.piecewise_survival"));
    let row = result.posterior_row().unwrap();
    assert_eq!(row.estimator, "bayesian.piecewise_survival");
    assert_eq!((row.uncertainty.as_str(), row.calibration.as_str()), ("posterior", "unmeasured"));
    assert!(!row.licensed);
    assert_eq!(row.coordinates, ["survival_difference", "rmst_difference"]);
    // Events and exposure by hand; Gamma(1, 1) updates are shape 1 + events, rate 1 + exposure.
    assert_eq!(row.vectors["control_events"], [2.0, 1.0, 1.0]);
    assert_eq!(row.vectors["control_shape"], [3.0, 2.0, 2.0]);
    for (got, want) in row.vectors["control_rate"].iter().zip([6.4, 4.0, 2.5]) {
        close(*got, want, 1e-12);
    }
    // Closed-form means: E[exp(-h l)] = (rate / (rate + l))^shape per interval, and the
    // restricted means by quadrature of the mean survival curve.
    close(row.extras["control_survival_mean"], 0.211_236_355_241_364_2, 1e-12);
    close(row.extras["treated_survival_mean"], 0.447_326_240_329_155_6, 1e-12);
    close(row.mean[0], 0.236_089_885_087_791_4, 1e-12);
    close(row.extras["control_rmst_mean"], 1.619_688_017_350_5, 1e-9);
    close(row.extras["treated_rmst_mean"], 2.157_706_276_225_9, 1e-9);
    close(row.mean[1], 0.538_018_258_905_1, 1e-9);
    // The intervals and the probability come from a two-million-draw numpy simulation.
    close(row.lower[0], -0.191_49, 0.03);
    close(row.upper[0], 0.643_14, 0.03);
    close(row.lower[1], -0.401_23, 0.06);
    close(row.upper[1], 1.419_53, 0.06);
    close(row.extras["probability_treated_survives_longer"], 0.862_216, 0.012);

    // The joint covariance across the two contrasts is retained and symmetric, and agrees
    // with the retained draws of the posterior itself.
    let covariance = row.covariance.as_ref().unwrap();
    assert_eq!(covariance.len(), 4);
    close(covariance[1], covariance[2], 1e-12);
    assert!(covariance[0] > 0.0 && covariance[3] > 0.0);
    let posterior = result.posterior.as_ref().unwrap();
    assert_eq!(posterior.draws.n_draws, 40_000);
    assert_eq!(row.draws, 40_000);
}

#[test]
fn flat_prior_survival_posterior_tracks_the_frequentist_product_limit_on_the_same_rows() {
    // Deterministic exponential quantiles with hazards 0.4 and 0.25 and administrative
    // censoring at 3, so the piecewise-constant model is correct. A weak prior must land at the
    // frequentist product limit and restricted mean of the same rows.
    let n = 400;
    let (mut duration, mut event, mut treatment) = (Vec::new(), Vec::new(), Vec::new());
    for arm in [false, true] {
        let hazard = if arm { 0.25 } else { 0.4 };
        for i in 0..n {
            let u = (f64::from(i) + 0.5) / f64::from(n);
            let time = -(1.0 - u).ln() / hazard;
            duration.push(time.min(3.0));
            event.push(f64::from(u8::from(time <= 3.0)));
            treatment.push(f64::from(u8::from(arm)));
        }
    }
    let data = survival_table(&duration, &event, &treatment);
    let query = CausalQuery::Survival(survival_query(3.0, SurvivalFunctional::SurvivalAndRmst));
    let (frequentist, _, _) = run(&data, query.clone(), InferenceMode::Frequentist, 3).unwrap();
    let curve = frequentist.survival().unwrap();
    let cuts = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    let (posterior, _, _) =
        run(&data, query, bayesian(2_000, hazards(&cuts, 1e-3, 1e-3)), 3).unwrap();
    let row = posterior.posterior_row().unwrap();
    let survival_difference = curve.treated.last().unwrap() - curve.control.last().unwrap();
    close(row.mean[0], survival_difference, 0.02);
    close(row.mean[1], curve.rmst_treated.unwrap() - curve.rmst_control.unwrap(), 0.04);
    assert!(row.lower[0] > 0.0, "{:?}", row.lower);
    assert!(row.extras["probability_treated_survives_longer"] > 0.99);
}

#[test]
fn survival_posterior_artifact_round_trips_and_refuses_forgeries() {
    let (duration, event, treatment) = small_survival();
    let data = survival_table(&duration, &event, &treatment);
    let query = CausalQuery::Survival(survival_query(3.0, SurvivalFunctional::SurvivalAndRmst));
    let (result, prepared, ctx) =
        run(&data, query, bayesian(2_000, hazards(&[1.0, 2.0, 3.0], 1.0, 1.0)), 5).unwrap();
    let bytes = prepared.encode_contracted_result(&result, "bayesian-survival", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.posterior_row.as_ref(), result.posterior_row());
    assert!(body.survival.is_none() && body.posterior_artifact.is_some());
    assert!(body.estimate.is_none() && body.interval_lower.is_none());

    let encode = |wire: &antecedent_io::AnalysisResultWire| {
        antecedent_io::encode_analysis_result_artifact(
            wire,
            header.variable_names.clone(),
            "forged-posterior-row",
        )
    };
    let mut licensed = body.clone();
    licensed.posterior_row.as_mut().unwrap().licensed = true;
    assert_refused(
        encode(&licensed),
        "a posterior row must have supported licensing semantics, unmeasured calibration and aligned finite arrays",
        "licensed",
    );
    let mut calibrated = body.clone();
    calibrated.posterior_row.as_mut().unwrap().calibration = "measured".into();
    assert_refused(
        encode(&calibrated),
        "a posterior row must have supported licensing semantics, unmeasured calibration and aligned finite arrays",
        "calibrated",
    );
    let mut misaligned = body.clone();
    misaligned.posterior_row.as_mut().unwrap().covariance = Some(vec![0.0; 3]);
    assert_refused(
        encode(&misaligned),
        "a posterior row must have supported licensing semantics, unmeasured calibration and aligned finite arrays",
        "misaligned",
    );
    let mut inverted = body.clone();
    let row = inverted.posterior_row.as_mut().unwrap();
    std::mem::swap(&mut row.lower, &mut row.upper);
    assert_refused(
        encode(&inverted),
        "a posterior row must have supported licensing semantics, unmeasured calibration and aligned finite arrays",
        "inverted",
    );
    let mut without_posterior = body.clone();
    without_posterior.posterior_artifact = None;
    assert_refused(
        encode(&without_posterior),
        "a posterior row is attached to a query without its Bayesian counterpart",
        "without posterior artifact",
    );
    let mut beside_frequentist = body;
    beside_frequentist.survival = Some(frequentist_section());
    assert_refused(
        encode(&beside_frequentist),
        "a posterior row is attached to a query without its Bayesian counterpart",
        "beside frequentist section",
    );
}

/// A minimal frequentist section, to prove a posterior row never rides beside one.
fn frequentist_section() -> antecedent_io::SurvivalWire {
    antecedent_io::SurvivalWire {
        times: vec![0.0, 3.0],
        control: vec![1.0, 0.5],
        treated: vec![1.0, 0.5],
        rmst_control: Some(2.0),
        rmst_treated: Some(2.0),
        target_cause: None,
        tau: 3.0,
        minimum_event_risk_set: Some(1),
        uncertainty: "point_only_no_interval".into(),
        rmst_difference_interval: None,
        difference_at_tau_interval: None,
        bootstrap_replicates_requested: None,
        bootstrap_replicates_ok: None,
        assignment_counts: Some([6, 6]),
        graphless_support_status: None,
        censoring_survival_provenance: None,
        difference_band: None,
        band_unavailable_reason: None,
    }
}

#[test]
fn survival_posterior_refusals_are_typed_and_name_the_next_attempt() {
    let (duration, mut event, treatment) = small_survival();
    let query = |functional| CausalQuery::Survival(survival_query(3.0, functional));
    let model = || bayesian(1_000, hazards(&[1.0, 2.0, 3.0], 1.0, 1.0));
    let refusal = |error: antecedent::CausalError| {
        (error.reason_code().map(str::to_owned), error.to_string())
    };

    // Competing causes in the data.
    event[0] = 2.0;
    let data = survival_table(&duration, &event, &treatment);
    let (code, text) =
        refusal(run(&data, query(SurvivalFunctional::SurvivalAndRmst), model(), 1).unwrap_err());
    assert_eq!(code.as_deref(), Some("route_not_supported"));
    assert!(text.contains("survival_posterior.competing_causes_unsupported"), "{text}");
    assert!(text.contains("cause-specific") && text.contains("Next attempt"), "{text}");

    // A cumulative-incidence query is a competing-risk functional.
    let (_, event, _) = small_survival();
    let data = survival_table(&duration, &event, &treatment);
    let (_, text) = refusal(
        run(&data, query(SurvivalFunctional::CumulativeIncidence { target_cause: 1 }), model(), 1)
            .unwrap_err(),
    );
    assert!(text.contains("survival_posterior.competing_causes_unsupported"), "{text}");

    // The partition and prior are declared, and end at the query's horizon.
    let undeclared = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(1_000));
    let (code, text) =
        refusal(run(&data, query(SurvivalFunctional::SurvivalAndRmst), undeclared, 1).unwrap_err());
    assert_eq!(code.as_deref(), Some("option_not_applicable"));
    assert!(text.contains("survival_posterior.model_not_declared"), "{text}");
    assert!(text.contains("Next attempt"), "{text}");
    let short = bayesian(1_000, hazards(&[1.0, 2.0], 1.0, 1.0));
    let (_, text) =
        refusal(run(&data, query(SurvivalFunctional::SurvivalAndRmst), short, 1).unwrap_err());
    assert!(text.contains("survival_posterior.horizon_mismatch"), "{text}");

    // A backend other than the conjugate built-in model is not read by this posterior.
    let laplace = InferenceMode::Bayesian(
        BayesianConfig::laplace().n_draws(1_000).design_model(hazards(&[1.0, 2.0, 3.0], 1.0, 1.0)),
    );
    let (code, _) =
        refusal(run(&data, query(SurvivalFunctional::SurvivalAndRmst), laplace, 1).unwrap_err());
    assert_eq!(code.as_deref(), Some("option_not_applicable"));
}

// ---------------------------------------------------------------------------------------------
// Fixed group-dose policy
// ---------------------------------------------------------------------------------------------

fn dose_table(outcome: &[f64], dose: &[f64]) -> TabularData {
    let density = vec![1.0; outcome.len()];
    TabularData::from_f64_columns([("outcome", outcome), ("dose", dose), ("density", &density[..])])
        .unwrap()
}

fn dose_query(
    groups: Vec<Arc<str>>,
    bandwidth: f64,
    min_local_support: usize,
    policy: [(&str, f64); 2],
    reference: [(&str, f64); 2],
) -> ContinuousDoseResponseQuery {
    let rule = |pairs: [(&str, f64); 2]| -> Arc<[(Arc<str>, f64)]> {
        pairs.iter().map(|(group, dose)| (Arc::<str>::from(*group), *dose)).collect()
    };
    ContinuousDoseResponseQuery {
        outcome: VariableId::from_raw(0),
        dose: VariableId::from_raw(1),
        dose_density: VariableId::from_raw(2),
        baseline_groups: groups.into(),
        target_doses: Arc::from([]),
        bandwidth,
        min_local_support,
        density_provenance: Arc::from("known"),
        fixed_policy: Some(FixedGroupDosePolicy {
            policy_doses: rule(policy),
            reference_doses: rule(reference),
        }),
    }
}

fn polynomial(
    support: (f64, f64),
    scales: &[f64],
    residual: ResidualPrior,
) -> DesignPosteriorModel {
    DesignPosteriorModel::DosePolynomial {
        support,
        coefficient_scales: Arc::from(scales),
        residual,
    }
}

/// Two groups of eight rows (the per-group ridge reference fixture).
fn two_group_rows() -> (Vec<f64>, Vec<f64>, Vec<Arc<str>>) {
    let mut dose = vec![0.2, 0.6, 1.1, 1.5, 1.9, 2.4, 2.8, 3.3];
    let mut outcome = vec![1.0, 1.6, 2.2, 2.5, 2.4, 2.0, 1.7, 1.1];
    let mut groups: Vec<Arc<str>> = vec![Arc::from("A"); 8];
    dose.extend([0.4, 0.9, 1.3, 1.8, 2.2, 2.6, 3.0, 3.7]);
    outcome.extend([0.5, 0.9, 1.0, 1.4, 1.9, 2.0, 2.4, 2.9]);
    groups.extend(vec![Arc::<str>::from("B"); 8]);
    (outcome, dose, groups)
}

#[test]
fn dose_policy_posterior_matches_the_per_group_conjugate_reference() {
    let (outcome, dose, groups) = two_group_rows();
    let data = dose_table(&outcome, &dose);
    let query = dose_query(groups, 1.0, 2, [("A", 1.5), ("B", 2.5)], [("A", 2.0), ("B", 1.0)]);
    let model = polynomial((0.0, 4.0), &[3.0, 2.0, 1.0], ResidualPrior::KnownVariance(0.25));
    let (result, _, _) =
        run(&data, CausalQuery::ContinuousDoseResponse(query), bayesian(40_000, model), 5).unwrap();

    assert!(result.continuous_dose_response().is_none());
    assert_eq!(result.support_status(), None);
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("bayesian.continuous_dose_policy"));
    let row = result.posterior_row().unwrap();
    assert_eq!(row.estimator, "bayesian.continuous_dose_policy");
    assert!(!row.licensed);
    assert_eq!(row.coordinates, ["policy_value", "reference_value", "incremental_value"]);
    // Per-group ridge posteriors from numpy and scipy adaptive quadrature of the window
    // average (not the Gauss-Legendre rule the implementation uses).
    let exact = row.exact_mean.as_ref().unwrap();
    for (got, want) in exact.iter().zip([1.915_595_66, 1.431_440_23, 0.484_155_43]) {
        close(*got, want, 1e-7);
    }
    for (got, want) in
        row.vectors["exact_variances"].iter().zip([0.019_981_54, 0.023_910_98, 0.018_089_61])
    {
        close(*got, want, 1e-7);
    }
    for g in 0..3 {
        close(row.mean[g], exact[g], 0.01);
    }
    // The retained joint draws: paired policy and reference draws give the incremental
    // variance, which independent draws would overstate (0.0199 + 0.0239 = 0.0438).
    let covariance = row.covariance.as_ref().unwrap();
    close(covariance[8], 0.018_089_61, 0.002);
    close(covariance[0], 0.019_981_54, 0.002);
    close(covariance[8], covariance[0] + covariance[4] - 2.0 * covariance[1], 1e-9);
    assert!(row.extras["probability_incremental_positive"] > 0.99);
    assert_eq!(row.extras["minimum_local_rows"], 4.0);
    let posterior = result.posterior.as_ref().unwrap();
    assert_eq!(posterior.draws.n_draws, 40_000);
}

#[test]
fn dose_policy_posterior_tracks_the_frequentist_kernel_row_on_the_same_rows() {
    let rows = 800_usize;
    let dose: Vec<f64> = (0..rows).map(|i| ((i * 37) % 997) as f64 / 997.0).collect();
    let outcome: Vec<f64> = (0..rows)
        .map(|i| {
            1.0 + f64::from(u8::from(i >= rows / 2))
                + 0.35 * (0.13 * i as f64).sin()
                + if i < rows / 2 { 1.5 * dose[i] } else { 2.0 * dose[i] }
        })
        .collect();
    let groups: Vec<Arc<str>> =
        (0..rows).map(|i| Arc::<str>::from(if i < rows / 2 { "a" } else { "b" })).collect();
    let data = dose_table(&outcome, &dose);
    let query = dose_query(groups, 0.2, 80, [("a", 0.7), ("b", 0.8)], [("a", 0.3), ("b", 0.4)]);
    let (frequentist, _, _) = run(
        &data,
        CausalQuery::ContinuousDoseResponse(query.clone()),
        InferenceMode::Frequentist,
        7,
    )
    .unwrap();
    let value = frequentist.continuous_dose_response().unwrap().fixed_policy.as_ref().unwrap();
    let model = polynomial(
        (0.0, 1.1),
        &[3.0, 2.0, 1.0],
        ResidualPrior::InverseGamma { shape: 2.0, scale: 0.1 },
    );
    let (posterior, _, _) =
        run(&data, CausalQuery::ContinuousDoseResponse(query), bayesian(4_000, model), 7).unwrap();
    let row = posterior.posterior_row().unwrap();
    // The linear truth makes the kernel-smoothed incremental value exactly 0.7 in each group's
    // symmetric window: 0.5 * 1.5 * 0.4 + 0.5 * 2.0 * 0.4.
    close(row.mean[2], 0.7, 0.015);
    close(row.mean[2], value.incremental_value, 0.015);
    close(row.mean[0], value.policy_value, 0.015);
    close(row.mean[1], value.reference_value, 0.015);
    assert!(row.exact_mean.is_none(), "unknown variance has no exact Gaussian moments");
    assert!(row.lower[2] < 0.7 + 0.1 && 0.7 - 0.1 < row.upper[2]);
}

#[test]
fn dose_policy_artifact_round_trips_and_refuses_forgeries() {
    let (outcome, dose, groups) = two_group_rows();
    let data = dose_table(&outcome, &dose);
    let query = dose_query(groups, 1.0, 2, [("A", 1.5), ("B", 2.5)], [("A", 2.0), ("B", 1.0)]);
    let model = polynomial((0.0, 4.0), &[3.0, 2.0, 1.0], ResidualPrior::KnownVariance(0.25));
    let (result, prepared, ctx) =
        run(&data, CausalQuery::ContinuousDoseResponse(query), bayesian(2_000, model), 5).unwrap();
    let bytes = prepared.encode_contracted_result(&result, "bayesian-dose-policy", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.posterior_row.as_ref(), result.posterior_row());
    assert!(body.continuous_dose_response.is_none() && body.posterior_artifact.is_some());
    let mut forged = body.clone();
    forged.posterior_row.as_mut().unwrap().licensed = true;
    assert_refused(
        antecedent_io::encode_analysis_result_artifact(&forged, header.variable_names.clone(), "x"),
        "a posterior row must have supported licensing semantics, unmeasured calibration and aligned finite arrays",
        "licensed",
    );
    let mut missing = body;
    missing.posterior_row = None;
    assert_refused(
        antecedent_io::encode_analysis_result_artifact(&missing, header.variable_names, "y"),
        "continuous-dose query and result section must appear together",
        "missing posterior row",
    );
}

#[test]
fn dose_policy_refusals_are_typed_and_name_the_next_attempt() {
    let (outcome, dose, groups) = two_group_rows();
    let data = dose_table(&outcome, &dose);
    let query =
        || dose_query(groups.clone(), 1.0, 2, [("A", 1.5), ("B", 2.5)], [("A", 2.0), ("B", 1.0)]);
    let model = || {
        bayesian(
            1_000,
            polynomial((0.0, 4.0), &[3.0, 2.0, 1.0], ResidualPrior::KnownVariance(0.25)),
        )
    };
    let refusal = |error: antecedent::CausalError| {
        (error.reason_code().map(str::to_owned), error.to_string())
    };

    let undeclared = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(1_000));
    let (code, text) = refusal(
        run(&data, CausalQuery::ContinuousDoseResponse(query()), undeclared, 1).unwrap_err(),
    );
    assert_eq!(code.as_deref(), Some("option_not_applicable"));
    assert!(text.contains("continuous_dose_policy.model_not_declared"), "{text}");
    assert!(text.contains("Next attempt"), "{text}");

    let mut grid = query();
    grid.target_doses = Arc::from([1.0]);
    let (_, text) =
        refusal(run(&data, CausalQuery::ContinuousDoseResponse(grid), model(), 1).unwrap_err());
    assert!(text.contains("continuous_dose_policy.grid_not_read"), "{text}");

    let mut grid_only = query();
    grid_only.fixed_policy = None;
    grid_only.target_doses = Arc::from([1.0]);
    let (code, text) = refusal(
        run(&data, CausalQuery::ContinuousDoseResponse(grid_only), model(), 1).unwrap_err(),
    );
    assert_eq!(code.as_deref(), Some("route_not_supported"));
    assert!(text.contains("continuous_dose_policy.policy_required"), "{text}");

    // A window that leaves the declared support is the estimator's own refusal.
    let mut wide = query();
    wide.bandwidth = 2.0;
    let (_, text) =
        refusal(run(&data, CausalQuery::ContinuousDoseResponse(wide), model(), 1).unwrap_err());
    assert!(text.contains("boundary kernels"), "{text}");
}

// ---------------------------------------------------------------------------------------------
// Randomized quantile treatment effect
// ---------------------------------------------------------------------------------------------

use antecedent::EstimatorId;
use antecedent_core::{AverageEffectQuery, OutcomeFunctional};
use antecedent_graph::{Dag, DenseNodeId};

fn quantile_grid(concentration: f64) -> DesignPosteriorModel {
    DesignPosteriorModel::QuantileGrid {
        thresholds: Arc::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        bin_concentration: concentration,
    }
}

/// Twelve control and twelve treated outcomes with the 0, 1, ..., 5 threshold grid.
fn quantile_table() -> TabularData {
    let control = [0.4, 0.8, 1.1, 1.3, 1.7, 2.2, 2.4, 2.9, 3.3, 3.6, 4.1, 4.6];
    let active = [0.9, 1.4, 1.8, 2.1, 2.6, 2.8, 3.1, 3.4, 3.9, 4.3, 4.4, 4.8];
    let mut outcome = control.to_vec();
    outcome.extend(active);
    let mut treatment = vec![0.0; control.len()];
    treatment.extend(vec![1.0; active.len()]);
    TabularData::from_f64_columns([("t", treatment.as_slice()), ("y", outcome.as_slice())]).unwrap()
}

fn quantile_query(tau: f64) -> AverageEffectQuery {
    let mut query =
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    query.outcome_functional = OutcomeFunctional::quantile(tau);
    query
}

fn run_quantile(
    data: &TabularData,
    query: AverageEffectQuery,
    model: Option<DesignPosteriorModel>,
    seed: u64,
) -> Result<(StudyResult, antecedent::PreparedStudy, ExecutionContext), antecedent::CausalError> {
    let mut dag = Dag::with_variables(2);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let mut config = BayesianConfig::conjugate().n_draws(40_000);
    if let Some(model) = model {
        config = config.design_model(model);
    }
    let ctx = ExecutionContext::for_tests(seed);
    let study = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::AverageEffect(query))
        .estimator(EstimatorId::BayesianRandomizedQuantile)
        .inference(InferenceMode::Bayesian(config))
        .refute(antecedent::RefuteSuite::None)
        .build()?;
    let prepared = study.prepare(&ctx)?;
    let result = prepared.estimate(data, &ctx)?;
    Ok((result, prepared, ctx))
}

#[test]
fn quantile_posterior_matches_the_dirichlet_reference_and_keeps_joint_draws() {
    let data = quantile_table();
    let (result, _, _) =
        run_quantile(&data, quantile_query(0.5), Some(quantile_grid(0.5)), 9).unwrap();
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("bayesian.randomized_quantile"));
    let row = result.posterior_row().unwrap();
    assert_eq!(row.estimator, "bayesian.randomized_quantile");
    assert!(!row.licensed && row.calibration == "unmeasured");
    assert_eq!(row.coordinates, ["control_quantile", "treated_quantile", "quantile_difference"]);
    // Bin counts by hand (control 0,2,3,3,2,2,0; treated 0,1,2,3,3,3,0) under a 0.5 prior per
    // bin give total mass 15.5 and these cumulative masses at the grid.
    for (got, want) in row.vectors["control_cdf_mean"]
        .iter()
        .zip([0.5, 3.0, 6.5, 10.0, 12.5, 15.0].map(|a| a / 15.5))
    {
        close(*got, want, 1e-12);
    }
    for (got, want) in row.vectors["treated_cdf_mean"]
        .iter()
        .zip([0.5, 2.0, 4.5, 8.0, 11.5, 15.0].map(|a| a / 15.5))
    {
        close(*got, want, 1e-12);
    }
    // Unresolved-draw accounting: every requested draw is counted.
    assert_eq!(row.extras["draws_requested"], 40_000.0);
    assert!(row.extras["draws_resolved"] >= 39_600.0);
    // The route's scalar posterior is the quantile difference over the same resolved draws, with
    // the two arm quantiles retained beside it.
    let posterior = result.posterior.as_ref().unwrap();
    assert_eq!(posterior.draws.schema.n_quantities(), 3);
    let effect = result.estimate().as_effect().unwrap();
    close(effect.ate, row.mean[2], 1e-9);
    let joint = row.covariance.as_ref().unwrap();
    // Independent arms under randomization: the control-treated covariance is near zero and the
    // difference variance is the sum of the arm variances.
    assert!(joint[1].abs() < 0.05 * (joint[0] * joint[4]).sqrt());
    close(joint[8], joint[0] + joint[4] - 2.0 * joint[1], 1e-9);
}

#[test]
fn vague_prior_quantile_posterior_tracks_the_empirical_piecewise_linear_quantile() {
    // Empirical grid distribution function of the control arm: 0, 2/12, 5/12, 8/12, 10/12, 1, so
    // the median inverts to 2 + (1/2 - 5/12) / (1/4) = 7/3; the treated arm reaches one half at 3.
    let data = quantile_table();
    let (result, _, _) =
        run_quantile(&data, quantile_query(0.5), Some(quantile_grid(0.01)), 4).unwrap();
    let row = result.posterior_row().unwrap();
    close(row.mean[0], 7.0 / 3.0, 0.03);
    close(row.mean[1], 3.0, 0.03);
    close(row.mean[2], 3.0 - 7.0 / 3.0, 0.03);
}

#[test]
fn quantile_posterior_artifact_round_trips_and_refuses_forgeries() {
    let data = quantile_table();
    let (result, prepared, ctx) =
        run_quantile(&data, quantile_query(0.5), Some(quantile_grid(0.5)), 9).unwrap();
    let bytes = prepared.encode_contracted_result(&result, "bayesian-quantile", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.posterior_row.as_ref(), result.posterior_row());
    assert!(body.posterior_artifact.is_some());
    let encode = |wire: &antecedent_io::AnalysisResultWire| {
        antecedent_io::encode_analysis_result_artifact(wire, header.variable_names.clone(), "x")
    };
    let mut licensed = body.clone();
    licensed.posterior_row.as_mut().unwrap().licensed = true;
    assert_refused(
        encode(&licensed),
        "a posterior row must have supported licensing semantics, unmeasured calibration and aligned finite arrays",
        "licensed",
    );
    let mut foreign = body.clone();
    foreign.posterior_row.as_mut().unwrap().estimator = "bayesian.piecewise_survival".into();
    assert_refused(
        encode(&foreign),
        "a posterior row is attached to a query without its Bayesian counterpart",
        "foreign estimator",
    );
    let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
    assert!(
        consumed.acceptance.unresolved.iter().any(|reason| {
            reason.as_ref() == "dependencies.checked_bayesian_quantile_operation"
        })
    );
}

#[test]
fn quantile_posterior_refusals_are_typed_and_name_the_next_attempt() {
    let data = quantile_table();
    let text = |error: antecedent::CausalError| {
        (error.reason_code().map(str::to_owned), error.to_string())
    };

    let (code, message) = text(run_quantile(&data, quantile_query(0.5), None, 1).unwrap_err());
    assert_eq!(code.as_deref(), Some("option_not_applicable"));
    assert!(message.contains("quantile_posterior.model_not_declared"), "{message}");
    assert!(message.contains("Next attempt"), "{message}");

    // A level the grid cannot bracket in more than one percent of the draws.
    let (_, message) =
        text(run_quantile(&data, quantile_query(0.999), Some(quantile_grid(0.5)), 1).unwrap_err());
    assert!(message.contains("quantile_posterior.unresolved_draws"), "{message}");
    assert!(message.contains("Next attempt"), "{message}");

    // A confounded treatment: the Dirichlet arm law carries no covariates.
    let z: Vec<f64> = (0..24).map(|i| f64::from(i % 3)).collect();
    let t = data.float64_values(VariableId::from_raw(0)).unwrap();
    let y = data.float64_values(VariableId::from_raw(1)).unwrap();
    let confounded = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let ctx = ExecutionContext::for_tests(1);
    let study = Study::tabular(confounded.clone())
        .graph(dag)
        .query(CausalQuery::AverageEffect(quantile_query(0.5)))
        .estimator(EstimatorId::BayesianRandomizedQuantile)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(1_000).design_model(quantile_grid(0.5)),
        ))
        .refute(antecedent::RefuteSuite::None)
        .build()
        .unwrap();
    let refused = study.prepare(&ctx).and_then(|p| p.estimate(&confounded, &ctx)).unwrap_err();
    assert!(
        refused.to_string().contains("quantile_posterior.adjustment_not_supported"),
        "{refused}"
    );
}

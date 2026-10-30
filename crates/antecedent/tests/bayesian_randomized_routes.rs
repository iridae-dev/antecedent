//! Bayesian arm contrasts and switchback effect of the randomized designs through the public
//! study facade. The two-arm Gaussian IPW pseudo-posterior is licensed with calibration
//! unmeasured; other constructions retain their separate scope. Each posterior is compared
//! with a reference computed outside the library (quadrature of the Beta and Student-t densities,
//! enumeration of the Beta-Binomial missing-count laws, exact conjugate moments, a whitened
//! least-squares derivation for the switchback), the retained artifact is read with an
//! independent loader, and every boundary is refused with its next attempt.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{
    BayesianConfig, CausalError, DesignPosteriorModel, EstimatorId, InferenceMode,
    RandomizedEffectQuery, Study, StudyResult,
};
use antecedent_core::{RandomizationDesign, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::randomized_posterior::{ArmBetaPrior, ArmOutcomeModel};
use antecedent_estimate::switchback_posterior::{
    CarryoverDeclaration, PeriodAdjustment, SequenceDependence, SwitchbackModel, SwitchbackPrior,
};

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!((actual - expected).abs() <= tolerance, "{what}: {actual} vs {expected}");
}

fn units(n: usize, prefix: &str) -> Vec<Arc<str>> {
    (0..n).map(|i| Arc::<str>::from(format!("{prefix}-{i}"))).collect()
}

fn uniform_arms() -> DesignPosteriorModel {
    DesignPosteriorModel::ArmOutcome {
        model: ArmOutcomeModel::Bernoulli(ArmBetaPrior { alpha: 1.0, beta: 1.0 }),
    }
}

fn bayes(model: Option<DesignPosteriorModel>) -> InferenceMode {
    let config = BayesianConfig::conjugate().n_draws(4000);
    InferenceMode::Bayesian(match model {
        Some(model) => config.design_model(model),
        None => config,
    })
}

fn run(
    data: &TabularData,
    query: RandomizedEffectQuery,
    inference: InferenceMode,
    estimator: Option<EstimatorId>,
) -> Result<(antecedent::PreparedStudy, StudyResult, ExecutionContext), CausalError> {
    let ctx = ExecutionContext::for_tests(0xB1A5);
    let mut builder = Study::tabular(data.clone()).query(query).inference(inference);
    if let Some(estimator) = estimator {
        builder = builder.estimator(estimator);
    }
    let prepared = builder.build()?.prepare(&ctx)?;
    let result = prepared.estimate(data, &ctx)?;
    Ok((prepared, result, ctx))
}

fn column(result: &StudyResult, k: usize) -> (f64, f64, f64, f64) {
    let posterior = result.posterior.as_ref().expect("a posterior");
    let s = &posterior.summaries;
    (s.mean[k], s.sd[k], s.q025[k], s.q975[k])
}

/// Treated: successes 3, failures 1 (4 units); control: successes 2, failures 4 (6 units).
const TRIAL: [(bool, f64); 10] = [
    (true, 1.0),
    (false, 1.0),
    (false, 0.0),
    (true, 1.0),
    (false, 0.0),
    (true, 0.0),
    (false, 1.0),
    (true, 1.0),
    (false, 0.0),
    (false, 0.0),
];

fn trial() -> (TabularData, RandomizedEffectQuery) {
    let outcomes: Vec<f64> = TRIAL.iter().map(|r| r.1).collect();
    let data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
    let query = RandomizedEffectQuery::with_design(
        RandomizationDesign::Complete { treated_units: 4 },
        VariableId::from_raw(0),
        TRIAL.iter().map(|r| r.0).collect::<Vec<_>>(),
        vec![0.4; 10],
        units(10, "u"),
        units(10, "o"),
        ("control", "treated"),
    );
    (data, query)
}

#[test]
fn the_default_bayesian_route_is_the_finite_population_posterior_of_the_design_based_target() {
    let (data, query) = trial();
    let (_, result, _) = run(&data, query, bayes(Some(uniform_arms())), None).unwrap();
    assert_eq!(
        result.logical_plan.estimator.as_deref(),
        Some("bayesian.randomized_finite_population")
    );
    assert_eq!(result.support_status(), None, "the Bayesian randomized route is unlicensed");
    assert!(result.randomized_effect().is_none());
    // Enumeration of BetaBinomial(6; 4, 2) and BetaBinomial(4; 3, 5) missing counts.
    let (mean, sd, lower, upper) = column(&result, 0);
    close(mean, 0.35, 5e-3, "finite-population mean");
    close(sd, 0.188_034_951_158, 6e-3, "finite-population sd");
    assert!(lower <= mean && mean <= upper);
    assert!(result.diagnostics.iter().any(|d| {
        d.code.as_ref() == "estimate.randomized.bayesian_arm_contrast.estimand"
            && d.message.contains("estimand=finite_population")
    }));
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "estimate.quasi.bayesian.unlicensed")
    );
}

#[test]
fn the_superpopulation_estimator_answers_the_other_estimand() {
    let (data, query) = trial();
    let (_, result, _) = run(
        &data,
        query,
        bayes(Some(uniform_arms())),
        Some(EstimatorId::BayesianRandomizedSuperpopulation),
    )
    .unwrap();
    assert_eq!(
        result.logical_plan.estimator.as_deref(),
        Some("bayesian.randomized_superpopulation")
    );
    // Beta(4, 2) minus Beta(3, 5): quadrature of the densities.
    let (mean, sd, lower, upper) = column(&result, 0);
    close(mean, 0.291_666_666_667, 5e-3, "superpopulation mean");
    close(sd, 0.240_390_720_313, 6e-3, "superpopulation sd");
    close(lower, -0.208_237_307_626, 1.5e-2, "2.5% quantile");
    close(upper, 0.719_027_029_937, 1.5e-2, "97.5% quantile");
    assert!(result.diagnostics.iter().any(|d| d.message.contains("estimand=superpopulation")));
}

fn four_arm_data() -> (Vec<f64>, Vec<usize>) {
    let successes = [3_usize, 5, 6, 9];
    let (mut y, mut arm) = (vec![], vec![]);
    for (cell, &s) in successes.iter().enumerate() {
        for r in 0..10 {
            y.push(if r < s { 1.0 } else { 0.0 });
            arm.push(cell);
        }
    }
    (y, arm)
}

#[test]
fn many_arms_report_every_arm_against_the_reference() {
    let (y, arms) = four_arm_data();
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    let query = RandomizedEffectQuery::with_design(
        RandomizationDesign::MultiArm {
            assignment: Arc::from(arms.clone()),
            probabilities: vec![vec![0.25; 4]; 40].into(),
            arms: ["control", "a", "b", "c"].map(Arc::<str>::from).into(),
        },
        VariableId::from_raw(0),
        arms.iter().map(|arm| *arm != 0).collect::<Vec<_>>(),
        vec![0.25; 40],
        units(40, "u"),
        units(40, "o"),
        ("control", "a"),
    );
    let (_, result, _) = run(
        &data,
        query,
        bayes(Some(uniform_arms())),
        Some(EstimatorId::BayesianRandomizedSuperpopulation),
    )
    .unwrap();
    // Beta(4, 8), Beta(6, 6), Beta(7, 5), Beta(10, 2) arm posteriors: exact moments and the
    // quadrature quantiles of each contrast against the control arm.
    let expected = [
        (0.166_666_666_667, 0.190_590_624_966),
        (0.25, 0.189_184_032_866),
        (0.5, 0.166_666_666_667),
    ];
    for (k, (mean, sd)) in expected.into_iter().enumerate() {
        let (got_mean, got_sd, lower, upper) = column(&result, k);
        close(got_mean, mean, 5e-3, "contrast mean");
        close(got_sd, sd, 6e-3, "contrast sd");
        assert!(lower < got_mean && got_mean < upper);
    }
}

#[test]
fn the_factorial_main_effects_and_interaction_match_the_exact_moments() {
    let (y, cell) = four_arm_data();
    let a: Vec<bool> = cell.iter().map(|c| *c >= 2).collect();
    let b: Vec<bool> = cell.iter().map(|c| c % 2 == 1).collect();
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    let query = RandomizedEffectQuery::with_design(
        RandomizationDesign::Factorial2x2 {
            second_factor_assignment: Arc::from(b),
            cell_counts: [10, 10, 10, 10],
            second_factor_arms: (Arc::from("b0"), Arc::from("b1")),
        },
        VariableId::from_raw(0),
        a,
        vec![0.5; 40],
        units(40, "u"),
        units(40, "o"),
        ("a0", "a1"),
    );
    let (_, result, _) = run(
        &data,
        query,
        bayes(Some(uniform_arms())),
        Some(EstimatorId::BayesianRandomizedSuperpopulation),
    )
    .unwrap();
    // (exact mean, exact sd) of the A main effect, the B main effect and the interaction.
    let expected = [
        (0.291_666_666_667, 0.128_165_057_84),
        (0.208_333_333_333, 0.128_165_057_84),
        (0.083_333_333_333, 0.256_330_115_68),
    ];
    for (k, (mean, sd)) in expected.into_iter().enumerate() {
        let (got_mean, got_sd, _, _) = column(&result, k);
        close(got_mean, mean, 5e-3, "factorial mean");
        close(got_sd, sd, 6e-3, "factorial sd");
    }
}

#[test]
fn the_retained_artifact_carries_the_posterior_and_an_independent_loader_recomputes_it() {
    let (data, query) = trial();
    let (prepared, result, ctx) = run(&data, query, bayes(Some(uniform_arms())), None).unwrap();
    let (mean, ..) = column(&result, 0);
    let bytes = prepared.encode_contracted_result(&result, "bayes-randomized", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert!(body.randomized_effect.is_none());
    let posterior_bytes = body.posterior_artifact.clone().expect("the posterior is retained");
    let (wire, draws) = antecedent_io::decode_causal_posterior_bytes(&posterior_bytes).unwrap();
    assert_eq!(wire.n_draws, 4000);
    let first: &[f64] = &draws[..wire.n_draws as usize];
    let recomputed = first.iter().sum::<f64>() / first.len() as f64;
    close(recomputed, mean, 1e-9, "mean recomputed from the retained draws");
    // A corrupted posterior payload is refused. Dropping the section cannot be caught at
    // the wire layer, which does not know the inference mode; the contract section binds it.
    let mut corrupted = body;
    corrupted.posterior_artifact = Some(vec![1, 2, 3]);
    assert_eq!(
        antecedent_io::encode_analysis_result_artifact(&corrupted, header.variable_names, "c")
            .unwrap_err()
            .to_string(),
        "io error: failed to fill whole buffer"
    );
}

#[test]
fn the_boundaries_are_refused_with_their_next_attempt() {
    // No declared prior: a prior alone does not identify a contrast and none is defaulted.
    let (data, query) = trial();
    let error = run(&data, query, bayes(None), None).err().expect("refused");
    assert_eq!(error.reason_code(), Some("required_option_missing"));
    assert!(error.to_string().contains("design_model"), "{error}");

    // Unit-varying propensity.
    let (data, mut query) = trial();
    query.assignment_probabilities =
        (0..10).map(|i| if i == 3 { 0.3 } else { 0.4 }).collect::<Vec<_>>().into();
    let error = run(&data, query, bayes(Some(uniform_arms())), None).err().expect("refused");
    let text = error.to_string();
    // The randomized-effect query itself refuses unit-varying assignment probabilities.
    assert!(
        text.contains("complete assignment probabilities must equal treated_units / n"),
        "{text}"
    );

    // A cluster design has no arm-exchangeable likelihood here.
    let (data, mut query) = trial();
    query.design = RandomizationDesign::Cluster { treated_clusters: 4 };
    let error = run(&data, query, bayes(Some(uniform_arms())), None).err().expect("refused");
    assert_eq!(error.reason_code(), Some("route_not_supported"));
    assert!(
        error
            .to_string()
            .contains("does not read a cluster, stratified or switchback assignment design"),
        "{error}"
    );

    // The design-based estimator does not run under Bayesian inference.
    let (data, query) = trial();
    let error = run(&data, query, bayes(Some(uniform_arms())), Some(EstimatorId::RandomizedNeyman))
        .err()
        .expect("refused");
    assert!(error.to_string().contains("estimator_inference_mismatch"), "{error}");
}

const ASSIGNMENT: [[bool; 4]; 4] = [
    [true, false, true, false],
    [false, true, false, true],
    [true, true, false, false],
    [false, false, true, true],
];
const OUTCOME: [[f64; 4]; 4] =
    [[2.3, 1.1, 2.9, 0.8], [0.7, 2.2, 1.4, 2.6], [2.8, 3.1, 0.9, 1.3], [1.2, 0.6, 2.4, 3.0]];

fn switchback() -> (TabularData, RandomizedEffectQuery) {
    let (mut y, mut a, mut sequences, mut periods) = (vec![], vec![], vec![], vec![]);
    for s in 0..4 {
        for t in 0..4 {
            y.push(OUTCOME[s][t]);
            a.push(ASSIGNMENT[s][t]);
            sequences.push(Arc::<str>::from(format!("s{s}")));
            periods.push(Arc::<str>::from(format!("p{t}")));
        }
    }
    let query = RandomizedEffectQuery::with_design(
        RandomizationDesign::Switchback { periods: periods.into() },
        VariableId::from_raw(0),
        a,
        vec![0.5; 16],
        sequences,
        units(16, "row"),
        ("off", "on"),
    );
    (TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap(), query)
}

fn switchback_model(
    carryover: CarryoverDeclaration,
    dependence: SequenceDependence,
) -> DesignPosteriorModel {
    DesignPosteriorModel::Switchback {
        model: SwitchbackModel {
            carryover,
            dependence,
            periods: PeriodAdjustment::None,
            prior: SwitchbackPrior {
                effect_precision: 0.5,
                nuisance_precision: 0.01,
                variance_shape: 2.0,
                variance_scale: 1.0,
            },
        },
    }
}

#[test]
fn the_switchback_posterior_matches_the_whitened_least_squares_reference() {
    let (data, query) = switchback();
    let model = switchback_model(
        CarryoverDeclaration::None,
        SequenceDependence::CompoundSymmetric { rho: 0.4 },
    );
    let (_, result, _) = run(&data, query, bayes(Some(model)), None).unwrap();
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("bayesian.switchback_effect"));
    assert_eq!(result.support_status(), None);
    // Student-t with 20 degrees of freedom: location, scale and quantiles from scipy.stats.t.
    let (mean, sd, lower, upper) = column(&result, 0);
    close(mean, 1.547_248_493, 1e-6, "direct mean");
    close(sd, 0.204_015_378_4, 2e-4, "direct sd");
    close(lower, 1.143_518_649, 2e-3, "direct 2.5%");
    close(upper, 1.950_978_337, 2e-3, "direct 97.5%");
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, antecedent_core::Assumption::Custom { id, .. }
        if id.as_ref() == "switchback_no_carryover"
    )));
}

#[test]
fn a_declared_first_order_carryover_adds_the_carryover_and_sustained_effects() {
    let (data, query) = switchback();
    let model = switchback_model(
        CarryoverDeclaration::FirstOrder,
        SequenceDependence::Autoregressive { rho: 0.5 },
    );
    let (_, result, _) = run(&data, query, bayes(Some(model)), None).unwrap();
    close(column(&result, 0).0, 1.570_684_728, 5e-3, "direct");
    close(column(&result, 1).0, 0.154_196_445, 5e-3, "carryover");
    close(column(&result, 2).0, 1.724_881_173, 5e-3, "sustained");
    let assumptions = &result.identification.required_assumptions.entries;
    assert!(assumptions.iter().any(|record| matches!(
        &record.assumption, antecedent_core::Assumption::Custom { id, .. }
        if id.as_ref() == "switchback_first_order_carryover"
    )));
    assert!(!assumptions.iter().any(|record| matches!(
        &record.assumption, antecedent_core::Assumption::Custom { id, .. }
        if id.as_ref() == "switchback_no_carryover"
    )));
}

#[test]
fn unrestricted_carryover_and_an_unspecified_dependence_are_typed_refusals() {
    let (data, query) = switchback();
    let error = run(
        &data,
        query,
        bayes(Some(switchback_model(
            CarryoverDeclaration::Unrestricted,
            SequenceDependence::Independent,
        ))),
        None,
    )
    .err()
    .expect("refused");
    assert_eq!(error.reason_code(), Some("cell_not_licensed"));
    assert!(error.to_string().contains("switchback_posterior.carryover_unrestricted"));
    assert!(error.to_string().contains("washout"));

    let (data, query) = switchback();
    let error = run(
        &data,
        query,
        bayes(Some(switchback_model(CarryoverDeclaration::None, SequenceDependence::Unspecified))),
        None,
    )
    .err()
    .expect("refused");
    assert_eq!(error.reason_code(), Some("cell_not_licensed"));
    assert!(error.to_string().contains("switchback_posterior.dependence_undeclared"));
    assert!(error.to_string().contains("sensitivity"));

    // No declaration at all.
    let (data, query) = switchback();
    let error = run(&data, query, bayes(None), None).err().expect("refused");
    assert_eq!(error.reason_code(), Some("required_option_missing"));
}

/// Alternating assignment at probability one half; 70% of units are compliers with a constant
/// effect of two, so the Wald ratio recovers two.
fn encouragement(n: usize, receipt_unrelated: bool) -> (TabularData, RandomizedEffectQuery) {
    let assigned: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let received: Vec<bool> = (0..n)
        .map(|i| if receipt_unrelated { i % 3 == 0 } else { assigned[i] && (i % 10) < 7 })
        .collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| {
            let effect = if received[i] && !receipt_unrelated { 2.0 } else { 0.0 };
            1.0 + 0.1 * (i % 7) as f64 + effect
        })
        .collect();
    let data = TabularData::from_f64_columns([("y", outcome.as_slice())]).unwrap();
    let query = RandomizedEffectQuery::bernoulli_itt(
        VariableId::from_raw(0),
        assigned,
        vec![0.5; n],
        units(n, "account"),
        units(n, "row"),
        ("control", "encouraged"),
    )
    .with_received_treatment(received);
    (data, query)
}

#[test]
fn the_complier_posterior_recovers_the_known_effect_and_is_unlicensed() {
    let (data, query) = encouragement(2_000, false);
    let (prepared, result, ctx) = run(&data, query, bayes(None), None).unwrap();
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("bayesian.complier_effect"));
    assert_eq!(result.support_status(), None, "the Bayesian complier route is unlicensed");
    assert_eq!(prepared.retained_operation().name(), "randomized");
    let posterior = result.posterior.as_ref().unwrap();
    let column = posterior.effect_column().unwrap();
    close(posterior.summaries.mean[column], 2.0, 0.3, "complier posterior mean");
    assert!(posterior.summaries.q025[column] < 2.0 && 2.0 < posterior.summaries.q975[column]);
    let bytes = prepared.encode_contracted_result(&result, "bayes-complier", &ctx).unwrap();
    let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
    let acceptance = &consumed.acceptance;
    assert!(
        acceptance.recognized
            && acceptance.claim_present
            && acceptance.unresolved.iter().all(|reason| reason.starts_with("dependencies.")),
        "unresolved: {:?}",
        acceptance.unresolved,
    );
}

#[test]
fn a_first_stage_that_reaches_zero_is_refused_not_published_as_a_ratio() {
    let (data, query) = encouragement(200, true);
    let error = run(&data, query, bayes(None), None).unwrap_err().to_string();
    assert!(error.contains("weak first stage") && error.contains("Next attempt"), "{error}");
}

#[test]
fn the_complier_posterior_refuses_an_arm_contrast_estimator() {
    let (data, query) = encouragement(400, false);
    let error = run(
        &data,
        query,
        bayes(Some(uniform_arms())),
        Some(EstimatorId::BayesianRandomizedSuperpopulation),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("Next attempt"), "{error}");
}

mod dose_grid_route {
    //! The standalone Bayesian randomized dose-grid route (`bayesian.dose_grid`) through the
    //! facade: evaluated, exported, and consumed by a fresh reader that recomputes the
    //! posterior from the embedded table. The route has no `Study` query, so its retained
    //! artifact is the posterior-row artifact rather than an analysis-result body. The
    //! reference is the conjugate ridge posterior of the polynomial dose curve computed with
    //! numpy from the normal equations on the rescaled design, outside the library.
    #![allow(
        clippy::float_cmp,
        reason = "bit-identical replay of the stored posterior is the property under test"
    )]

    use antecedent::analysis::bayesian_posterior_row::{
        BayesianPosteriorRow, DOSE_GRID_POSTERIOR_ESTIMATOR, DoseGridPosteriorWire,
        PosteriorRowArtifactError, PosteriorRowConsumeLimits, PosteriorRowRequestWire,
        ResidualPriorWire,
    };
    use antecedent::prelude::ExecutionContext;
    use antecedent_io::bayesian_posterior_row_artifact::PosteriorRowArtifactWire;
    use antecedent_io::dose_grid_functional_artifact::DoseFunctionalWire;

    const ROUTE: &str = "bayesian.dose_grid";
    const DOSE: [f64; 10] = [0.1, 0.5, 0.9, 1.4, 1.8, 2.2, 2.7, 3.1, 3.5, 3.9];
    const OUTCOME: [f64; 10] = [1.2, 1.9, 2.1, 3.0, 3.2, 3.1, 3.0, 2.6, 2.2, 1.5];

    fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
        assert!((actual - expected).abs() <= tolerance, "{what}: {actual} vs {expected}");
    }

    fn request(kind: &str) -> PosteriorRowRequestWire {
        let contrast = kind == "contrast";
        PosteriorRowRequestWire::DoseGrid(DoseGridPosteriorWire {
            design: "randomized_dose".into(),
            functional: DoseFunctionalWire {
                kind: kind.into(),
                from: contrast.then_some(1.0),
                to: contrast.then_some(3.0),
            },
            grid: if contrast { Vec::new() } else { vec![0.5, 2.0, 3.5] },
            support: (0.0, 4.0),
            coefficient_scales: vec![3.0, 2.0, 1.0],
            residual: ResidualPriorWire::KnownVariance(0.25),
            draws: 40_000,
            seed: 21,
            dose: DOSE.to_vec(),
            outcome: OUTCOME.to_vec(),
        })
    }

    fn live(kind: &str) -> BayesianPosteriorRow {
        let ctx = ExecutionContext::for_tests(0xD05E);
        BayesianPosteriorRow::evaluate_with_context(request(kind), &ctx).unwrap()
    }

    fn consume(bytes: &[u8]) -> Result<BayesianPosteriorRow, PosteriorRowArtifactError> {
        BayesianPosteriorRow::consume(bytes, PosteriorRowConsumeLimits::default())
    }

    #[test]
    fn the_bayesian_dose_grid_route_replays_to_the_conjugate_reference_through_a_fresh_consumer() {
        assert_eq!(ROUTE, DOSE_GRID_POSTERIOR_ESTIMATOR);
        let references: [(&str, &[&str], &[f64], &[f64]); 3] = [
            (
                "level",
                &["level(0.5)", "level(2)", "level(3.5)"],
                &[2.018_449_65, 2.714_270_59, 2.318_891_91],
                &[
                    0.064_536_70,
                    0.016_412_15,
                    -0.005_911_84,
                    0.016_412_15,
                    0.041_541_01,
                    0.015_904_12,
                    -0.005_911_84,
                    0.015_904_12,
                    0.063_660_80,
                ],
            ),
            (
                "derivative",
                &["derivative(0.5)", "derivative(2)", "derivative(3.5)"],
                &[0.827_613_84, 0.100_147_42, -0.627_319_00],
                &[],
            ),
            ("contrast", &["contrast(1->3)"], &[0.200_294_84], &[0.062_231_64]),
        ];
        for (kind, coordinates, mean, covariance) in references {
            let row = live(kind);
            assert_eq!(row.estimator(), ROUTE);
            let consumed = consume(&row.export().unwrap()).unwrap();
            // The consumed summaries are the live ones, bit for bit.
            assert_eq!(consumed.result(), row.result(), "{kind}: consumed equals live");
            assert_eq!(consumed.digests(), row.digests());
            let result = consumed.result();
            assert_eq!(result.coordinates, coordinates, "{kind}");
            assert!(!result.licensed);
            assert_eq!(result.calibration, "unmeasured");
            // And both equal the independent closed form.
            let exact = result.exact_mean.as_ref().unwrap();
            for (got, want) in exact.iter().zip(mean) {
                close(*got, *want, 1e-7, kind);
            }
            for (got, want) in result.mean.iter().zip(mean) {
                close(*got, *want, 0.01, kind);
            }
            if !covariance.is_empty() {
                let exact_covariance = result.exact_covariance.as_ref().unwrap();
                for (got, want) in exact_covariance.iter().zip(covariance) {
                    close(*got, *want, 1e-7, kind);
                }
                let width = coordinates.len();
                let joint = result.covariance.as_ref().unwrap();
                for g in 0..width {
                    for h in 0..width {
                        let scale = (covariance[g * width + g] * covariance[h * width + h]).sqrt();
                        close(
                            joint[g * width + h],
                            covariance[g * width + h],
                            0.05 * scale,
                            "joint covariance",
                        );
                    }
                }
            }
            for g in 0..coordinates.len() {
                assert!(result.lower[g] < result.mean[g] && result.mean[g] < result.upper[g]);
            }
        }
    }

    #[test]
    fn a_tampered_bayesian_dose_grid_artifact_is_refused_with_its_exact_typed_error() {
        let bytes = live("level").export().unwrap();
        let sealed = PosteriorRowArtifactWire::decode(&bytes).unwrap();
        let reseal = |mut wire: PosteriorRowArtifactWire| {
            wire.premises_digest = wire.expected_premises_digest().unwrap();
            wire.data_digest = wire.expected_data_digest().unwrap();
            wire.export().unwrap()
        };

        // An edited table entry without fresh digests breaks the data identity ...
        let mut edited = sealed.clone();
        let PosteriorRowRequestWire::DoseGrid(w) = &mut edited.request;
        w.outcome[3] += 0.5;
        assert_eq!(
            consume(&edited.export().unwrap()).unwrap_err(),
            PosteriorRowArtifactError::DataMismatch
        );
        // ... and the same edit resealed is caught by the replay.
        assert_eq!(
            consume(&reseal(edited)).unwrap_err(),
            PosteriorRowArtifactError::ResultMismatch
        );

        // A changed prior scale breaks the premises identity, and resealed the replay.
        let mut prior = sealed.clone();
        let PosteriorRowRequestWire::DoseGrid(w) = &mut prior.request;
        w.coefficient_scales[2] = 0.5;
        assert_eq!(
            consume(&prior.export().unwrap()).unwrap_err(),
            PosteriorRowArtifactError::PremisesMismatch
        );
        assert_eq!(consume(&reseal(prior)).unwrap_err(), PosteriorRowArtifactError::ResultMismatch);

        // A forged stored posterior mean, resealed.
        let mut forged = sealed.clone();
        forged.result.mean[0] += 0.001;
        assert_eq!(
            consume(&reseal(forged)).unwrap_err(),
            PosteriorRowArtifactError::ResultMismatch
        );

        // A claimed licence is refused as unsupported semantics before any replay.
        let mut licensed = sealed;
        licensed.result.licensed = true;
        assert!(matches!(
            consume(&reseal(licensed)).unwrap_err(),
            PosteriorRowArtifactError::UnsupportedSemantics(_)
        ));
        assert!(matches!(consume(b"not cbor"), Err(PosteriorRowArtifactError::Undecodable(_))));
    }
}
// ---------------------------------------------------------------- missing outcomes, weighted

/// 24 units alternating control (even rows) and treated (odd rows); the binary baseline `x`
/// flips every second pair. Nine outcomes are missing.
const IPW_OUTCOME: [f64; 24] = [
    1.2, 2.1, 0.7, 2.9, 1.8, 3.4, 0.9, 2.4, 1.5, 3.1, 2.2, 2.7, 1.1, 3.6, 0.4, 2.0, 1.9, 3.0, 1.3,
    3.8, 0.8, 2.5, 1.7, 3.3,
];
const IPW_MISSING: [usize; 9] = [1, 6, 9, 12, 15, 17, 20, 22, 23];

fn ipw_trial(design: RandomizationDesign) -> (TabularData, RandomizedEffectQuery) {
    let outcome: Vec<f64> =
        (0..24).map(|i| if IPW_MISSING.contains(&i) { f64::NAN } else { IPW_OUTCOME[i] }).collect();
    let baseline: Vec<f64> = (0..24).map(|i| f64::from(u8::from((i / 2) % 2 == 1))).collect();
    let data = TabularData::from_f64_columns([
        ("outcome", outcome.as_slice()),
        ("x", baseline.as_slice()),
    ])
    .unwrap();
    let query = RandomizedEffectQuery::with_design(
        design,
        VariableId::from_raw(0),
        (0..24).map(|i| i % 2 == 1).collect::<Vec<_>>(),
        vec![0.5; 24],
        units(24, "u"),
        units(24, "o"),
        ("control", "treated"),
    );
    (data, query)
}

fn gaussian_arms() -> DesignPosteriorModel {
    DesignPosteriorModel::ArmOutcome {
        model: ArmOutcomeModel::Gaussian(
            antecedent_estimate::randomized_posterior::ArmNormalPrior {
                mean: 0.0,
                pseudo_units: 1.0e-3,
                variance_shape: 1.0,
                variance_scale: 1.0,
            },
        ),
    }
}

fn run_weighted(
    data: &TabularData,
    query: RandomizedEffectQuery,
    inference: InferenceMode,
    estimator: Option<EstimatorId>,
    missing: Option<antecedent_data::MissingDeclaration>,
) -> Result<StudyResult, CausalError> {
    let ctx = ExecutionContext::for_tests(0xB1A6);
    let mut builder = Study::tabular(data.clone()).query(query).inference(inference);
    if let Some(estimator) = estimator {
        builder = builder.estimator(estimator);
    }
    if let Some(missing) = missing {
        builder = builder.missing(missing);
    }
    builder.build()?.prepare(&ctx)?.estimate(data, &ctx)
}

fn inverse_probability(covariates: &[u32]) -> Option<antecedent_data::MissingDeclaration> {
    Some(
        antecedent_data::MissingDeclaration::new(
            antecedent_data::MissingPolicy::InverseProbabilityOfObservation,
        )
        .with_covariates(covariates.iter().map(|&c| VariableId::from_raw(c)).collect::<Vec<_>>()),
    )
}

/// The weighted conjugate update is the Normal-Inverse-Gamma update on the inverse-probability
/// pseudo-outcomes `ybar_s + R (y - ybar_s) / p_s` of each response stratum `s` (the influence
/// function of the post-stratified responder mean, so the posterior spread carries the response
/// uncertainty). Stratifying by arm alone, then by arm and the binary baseline (a saturated
/// logistic observation model, whose fitted probabilities are the stratum response rates), the
/// contrast of two independent Student-t arm laws has mean `mn_1 - mn_0` and variance
/// `sum bn / (kn (an - 1))`. The numbers below come from an independent script
/// that applies that closed form to the table above.
#[test]
fn weighted_conjugate_update_matches_the_closed_form_pseudo_outcome_posterior() {
    let design = RandomizationDesign::Bernoulli;
    let superpopulation = Some(EstimatorId::BayesianRandomizedSuperpopulation);
    for (covariates, mean, sd) in [
        (&[][..], 1.667_718_166_343_280_7, 0.326_027_745_643_939_3),
        (&[1][..], 1.683_193_067_244_396_4, 0.321_509_057_609_117_6),
    ] {
        let (data, query) = ipw_trial(design.clone());
        let result = run_weighted(
            &data,
            query,
            bayes(Some(gaussian_arms())),
            superpopulation,
            inverse_probability(covariates),
        )
        .unwrap();
        let (got_mean, got_sd, lower, upper) = column(&result, 0);
        close(got_mean, mean, 5e-3, "weighted posterior mean");
        close(got_sd, sd, 6e-3, "weighted posterior sd");
        assert!(lower < got_mean && got_mean < upper);
        let report = result.missing_data().as_ref().expect("a missing-data report");
        assert_eq!(report.policy().name(), "inverse_probability");
        assert_eq!(report.accounting.rows_total, 24);
        assert_eq!(result.support_status(), Some(antecedent::CellStatus::Licensed));
        assert!(result.posterior_row().unwrap().licensed);
    }
}

#[test]
fn licensed_ipow_posterior_recovers_the_causal_target_and_declared_prior_update() {
    // In each arm, half the units have baseline z=0 and half z=1, with
    // Y(a)=1+2a+5z+e and balanced e=+-1. Response is 3/4 vs 1/4 by z in
    // control, reversed in treatment. Thus complete-case bias is 2.5, while
    // the post-stratified arm contrast is exactly the structural effect 2.
    for design in
        [RandomizationDesign::Bernoulli, RandomizationDesign::Complete { treated_units: 64 }]
    {
        for pseudo_units in [0.001, 5.0] {
            let mut y = Vec::new();
            let mut z = Vec::new();
            let mut assigned = Vec::new();
            let mut independent_pseudo = [Vec::new(), Vec::new()];
            let mut responder_sums = [0.0; 2];
            let mut responder_counts = [0; 2];
            for unit in 0..64 {
                let baseline = usize::from(unit >= 32);
                for arm in 0..2 {
                    let count = if arm == baseline { 24 } else { 8 };
                    let pi = count as f64 / 32.0;
                    let seen = unit % 32 < count;
                    let centre = 1.0 + 2.0 * arm as f64 + 5.0 * baseline as f64;
                    let noise = if unit % 2 == 0 { -1.0 } else { 1.0 };
                    y.push(if seen { centre + noise } else { f64::NAN });
                    z.push(baseline as f64);
                    assigned.push(arm == 1);
                    independent_pseudo[arm].push(centre + if seen { noise / pi } else { 0.0 });
                    if seen {
                        responder_sums[arm] += centre + noise;
                        responder_counts[arm] += 1;
                    }
                }
            }
            let naive = responder_sums[1] / responder_counts[1] as f64
                - responder_sums[0] / responder_counts[0] as f64;
            assert!((naive - 4.5).abs() < 1e-12);
            let data =
                TabularData::from_f64_columns([("outcome", y.as_slice()), ("z", z.as_slice())])
                    .unwrap();
            let query = RandomizedEffectQuery::with_design(
                design.clone(),
                VariableId::from_raw(0),
                assigned,
                vec![0.5; 128],
                units(128, "unit"),
                units(128, "outcome"),
                ("control", "treated"),
            );
            let prior = antecedent_estimate::randomized_posterior::ArmNormalPrior {
                mean: 0.0,
                pseudo_units,
                variance_shape: 2.0,
                variance_scale: 1.0,
            };
            // Definition-level NIG update on independently derived saturated-stratum
            // pseudo-outcomes, without either product fitting routine.
            let moments: Vec<(f64, f64)> = independent_pseudo
                .iter()
                .map(|values| {
                    let n = values.len() as f64;
                    let mean = values.iter().sum::<f64>() / n;
                    let kappa = prior.pseudo_units + n;
                    let shape = prior.variance_shape + n / 2.0;
                    let scale = prior.variance_scale
                        + 0.5 * values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                        + prior.pseudo_units * n * (mean - prior.mean).powi(2) / (2.0 * kappa);
                    (
                        (prior.pseudo_units * prior.mean + n * mean) / kappa,
                        scale / (kappa * (shape - 1.0)),
                    )
                })
                .collect();
            let result = run_weighted(
                &data,
                query,
                bayes(Some(DesignPosteriorModel::ArmOutcome {
                    model: ArmOutcomeModel::Gaussian(prior),
                })),
                Some(EstimatorId::BayesianRandomizedSuperpopulation),
                inverse_probability(&[1]),
            )
            .unwrap();
            assert_eq!(result.support_status(), Some(antecedent::CellStatus::Licensed));
            let (mean, sd, _, _) = column(&result, 0);
            close(mean, moments[1].0 - moments[0].0, 5e-3, "known-target posterior mean");
            close(sd, (moments[0].1 + moments[1].1).sqrt(), 6e-3, "known-target posterior spread");
            close(mean, 2.0 * 64.0 / (64.0 + pseudo_units), 5e-3, "declared prior shrinkage");
        }
    }
}

#[test]
fn the_weighted_posterior_is_not_the_responder_only_posterior() {
    // Responder means are 1.375 (control) and 3.042857 (treated): 1.667857. The covariate
    // model reweights the strata, so its posterior mean differs from that contrast by 0.015.
    let (data, query) = ipw_trial(RandomizationDesign::Bernoulli);
    let result = run_weighted(
        &data,
        query,
        bayes(Some(gaussian_arms())),
        Some(EstimatorId::BayesianRandomizedSuperpopulation),
        inverse_probability(&[1]),
    )
    .unwrap();
    let (mean, _, _, _) = column(&result, 0);
    assert!((mean - 1.667_857_142_857).abs() > 0.01, "{mean}");
}

#[test]
fn licensed_ipow_posterior_replays_after_the_builder_is_dropped() {
    use antecedent::analysis::RetainedOperation;
    for design in
        [RandomizationDesign::Bernoulli, RandomizationDesign::Complete { treated_units: 12 }]
    {
        let (data, query) = ipw_trial(design);
        let ctx = ExecutionContext::for_tests(0xB1A6);
        let prepare = || {
            Study::tabular(data.clone())
                .query(query.clone())
                .inference(bayes(Some(gaussian_arms())))
                .estimator(EstimatorId::BayesianRandomizedSuperpopulation)
                .missing(inverse_probability(&[1]).unwrap())
                .build()
                .unwrap()
                .prepare(&ctx)
                .unwrap()
        };
        let prepared = prepare();
        assert!(matches!(prepared.retained_operation(), RetainedOperation::Randomized));
        let result = prepared.estimate_retained(&ctx).unwrap();
        assert_eq!(result.support_status(), Some(antecedent::CellStatus::Licensed));
        close(column(&result, 0).0, 1.683_193_067_244_396_4, 5e-3, "retained IPW posterior mean");
        let bytes = prepared.encode_contracted_result(&result, "licensed-ipow", &ctx).unwrap();
        drop(prepared);
        let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
        assert!(consumed.acceptance.accepts_as_claim(), "{:?}", consumed.acceptance);
        let (accepted, receipt) =
            antecedent_io::accept_claim(&bytes, &antecedent_contract::ConsumerProfile::full())
                .unwrap();
        assert!(accepted.acceptance.accepts_as_claim());
        assert!(receipt.equivalent_claim());
        let (forwarded, receipt) = antecedent_io::accept_claim(
            &bytes,
            &antecedent_contract::ConsumerProfile::forwarding(),
        )
        .unwrap();
        assert!(!forwarded.acceptance.accepts_as_claim());
        assert!(!receipt.equivalent_claim());
        let bounds = antecedent_io::distribution_replay::ReplayLimits {
            max_operations: 0,
            ..Default::default()
        };
        assert!(antecedent_io::consume_analysis_result_within(&bytes, &bounds).is_err());
        let row = consumed.body.posterior_row.as_ref().unwrap();
        assert!(row.licensed);
        assert_eq!(row.calibration, "unmeasured");
        let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
        for edit in 0..3 {
            let mut changed = body.clone();
            let row = changed.posterior_row.as_mut().unwrap();
            match edit {
                0 => row.mean[0] += 0.1,
                1 => row.randomized_ipow.as_mut().unwrap().prior[0] += 0.1,
                _ => {
                    let y = row
                        .randomized_ipow
                        .as_mut()
                        .unwrap()
                        .outcomes
                        .iter_mut()
                        .flatten()
                        .next()
                        .unwrap();
                    *y += 1.0;
                }
            }
            // Re-encode a valid container: numerical replay must reject a changed result,
            // prior or source even after the outer container checksums have been replaced.
            let forged = antecedent_io::encode_analysis_result_artifact(
                &changed,
                header.variable_names.clone(),
                "edited",
            )
            .unwrap();
            let mut forged_bytes = Vec::new();
            forged.write_to(&mut forged_bytes).unwrap();
            assert!(antecedent_io::consume_analysis_result(&forged_bytes).is_err(), "edit {edit}");
        }
    }
}

#[test]
fn weighted_bayesian_boundaries_are_typed_refusals() {
    // No declared policy for a missing outcome: the policy is required, never defaulted.
    let (data, query) = ipw_trial(RandomizationDesign::Bernoulli);
    let error = run_weighted(&data, query, bayes(Some(gaussian_arms())), None, None)
        .err()
        .expect("refused");
    assert_eq!(error.reason_code(), Some("missing_policy_required"), "{error}");

    // The finite-population estimand needs the unobserved share of every arm, which the
    // weighted pseudo-outcomes do not describe.
    let (data, query) = ipw_trial(RandomizationDesign::Bernoulli);
    let error =
        run_weighted(&data, query, bayes(Some(gaussian_arms())), None, inverse_probability(&[]))
            .err()
            .expect("refused");
    assert_eq!(error.reason_code(), Some("route_not_supported"), "{error}");
    assert!(error.to_string().contains("superpopulation"), "{error}");

    // Beta-Bernoulli arms cannot take a weighted outcome.
    let (data, query) = ipw_trial(RandomizationDesign::Bernoulli);
    let error = run_weighted(
        &data,
        query,
        bayes(Some(uniform_arms())),
        Some(EstimatorId::BayesianRandomizedSuperpopulation),
        inverse_probability(&[]),
    )
    .err()
    .expect("refused");
    assert_eq!(error.reason_code(), Some("route_not_supported"), "{error}");
    assert!(error.to_string().contains("Normal-Inverse-Gamma"), "{error}");
}

#[test]
fn unlicensed_multi_arm_and_factorial_ipow_are_typed_refusals() {
    let (mut y, cells) = four_arm_data();
    for i in [1, 11, 21, 31] {
        y[i] = f64::NAN;
    }
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    for design in [
        RandomizationDesign::MultiArm {
            assignment: cells.clone().into(),
            probabilities: vec![vec![0.25; 4]; 40].into(),
            arms: ["control", "a", "b", "c"].map(Arc::<str>::from).into(),
        },
        RandomizationDesign::Factorial2x2 {
            second_factor_assignment: cells.iter().map(|c| c % 2 == 1).collect::<Vec<_>>().into(),
            cell_counts: [10; 4],
            second_factor_arms: (Arc::from("b0"), Arc::from("b1")),
        },
    ] {
        let multi = matches!(design, RandomizationDesign::MultiArm { .. });
        let query = RandomizedEffectQuery::with_design(
            design,
            VariableId::from_raw(0),
            cells.iter().map(|c| if multi { *c != 0 } else { *c >= 2 }).collect::<Vec<_>>(),
            vec![if multi { 0.25 } else { 0.5 }; 40],
            units(40, "u"),
            units(40, "o"),
            ("control", if multi { "a" } else { "treated" }),
        );
        let error = run_weighted(
            &data,
            query,
            bayes(Some(gaussian_arms())),
            Some(EstimatorId::BayesianRandomizedSuperpopulation),
            inverse_probability(&[]),
        )
        .err()
        .expect("unsupported IPW design must refuse");
        assert_eq!(error.reason_code(), Some("route_not_supported"), "{error}");
    }
}

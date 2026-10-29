//! Independent oracles for the uncertainty derived for adaptive procedures
//! whose units contribute several correlated measurements (repeated measures
//! with a declared within-unit correlation, and clustered outcomes with a
//! unit-level random effect).
//!
//! The oracles are written here, not taken from the code under test:
//!
//! * the variance of a unit's mean measurement under compound symmetry,
//!   `sigma^2 (1 + (m - 1) rho) / m`, so the design effect is
//!   `1 + (m - 1) rho`;
//! * a hand-summed autoregressive correlation matrix and the closed-form
//!   generalized-least-squares variance `sigma^2 (1 + rho) / (2 + (m - 2) (1 - rho))`;
//! * the variance `tau^2 + sigma^2 / m` of a cluster mean with a random
//!   effect;
//! * enumerated tiny cases of a fixed two-arm allocation: the binomial
//!   number of units per arm, the normal contrast given that number, and the
//!   mass on which an arm is empty;
//! * the martingale property of the adaptively weighted score (unbiased
//!   under any history-dependent policy), checked by simulation with its
//!   Monte Carlo standard error.
//!
//! Each family, outcome structure, analysis route and monitoring combination
//! is its own test: evidence for one cell is not evidence for another.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::adaptive_inference::{
    AdaptiveInferenceError as Error, AnalysisEstimator, ContractGap, DERIVED_UNMEASURED,
    FinalAnalysis, GaussianMixtureLaw, MAX_UNIT_LAG, OutcomeModelPrior, OutcomeStructure,
    RepeatedMeasuresDeclaration, RepeatedMeasuresRequest, RepeatedMeasuresSpec, Sidedness,
    StoppingRule, UncertaintyParadigm, UnitAssignment, UnitSummary, WeightScheme,
    WithinUnitCorrelation, admit_repeated_measures, derive_repeated_measures,
    exact_fixed_contrast_law, simulate_repeated_measures,
};

fn never() -> bool {
    false
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: expected {expected}, got {actual} (tolerance {tolerance})"
    );
}

fn compound(rho: f64) -> WithinUnitCorrelation {
    WithinUnitCorrelation::CompoundSymmetric { rho }
}

fn autoregressive(rho: f64) -> WithinUnitCorrelation {
    WithinUnitCorrelation::Autoregressive { rho }
}

fn repeated(
    measurements: u32,
    variance: f64,
    c: Option<WithinUnitCorrelation>,
) -> OutcomeStructure {
    OutcomeStructure::RepeatedMeasures { measurements, variance, correlation: c }
}

fn cluster_robust(summary: UnitSummary) -> AnalysisEstimator {
    AnalysisEstimator::ClusterRobust { summary }
}

fn weighted() -> AnalysisEstimator {
    AnalysisEstimator::AdaptivelyWeighted {
        summary: UnitSummary::Mean,
        scheme: WeightScheme::Uniform,
        prior: OutcomeModelPrior { mean: 0.0, strength: 1.0 },
        probability_floor: 0.05,
    }
}

fn even() -> UnitAssignment {
    UnitAssignment::Fixed { probabilities: vec![0.5, 0.5] }
}

fn greedy(epsilon: f64) -> UnitAssignment {
    UnitAssignment::EpsilonGreedy { epsilon, prior: OutcomeModelPrior { mean: 0.0, strength: 1.0 } }
}

fn declaration(
    assignment: UnitAssignment,
    enrolment: Vec<u32>,
    structure: OutcomeStructure,
    estimator: AnalysisEstimator,
) -> RepeatedMeasuresDeclaration {
    RepeatedMeasuresDeclaration {
        arms: 2,
        assignment: Some(assignment),
        enrolment: Some(enrolment),
        lag_looks: 0,
        structure: Some(structure),
        analysis: Some(FinalAnalysis { estimator, plus: 1, minus: 0 }),
        stopping: Some(StoppingRule::NoStopping),
    }
}

fn spec_of(d: RepeatedMeasuresDeclaration) -> RepeatedMeasuresSpec {
    RepeatedMeasuresSpec::new(d).expect("the declaration is valid")
}

fn gaps_of(d: RepeatedMeasuresDeclaration) -> Vec<ContractGap> {
    RepeatedMeasuresSpec::new(d).expect_err("the declaration is refused").gaps
}

fn premise_detail<T: std::fmt::Debug>(outcome: Result<T, Error>) -> &'static str {
    match outcome {
        Err(Error::PremiseFailed { detail, .. }) => detail,
        other => panic!("expected a premise failure, got {other:?}"),
    }
}

fn request(replications: u32, seed: u64) -> RepeatedMeasuresRequest {
    RepeatedMeasuresRequest { replications, seed }
}

// -- Closed forms of the unit summary -------------------------------------------------------------

#[test]
fn compound_symmetry_gives_the_design_effect_one_plus_m_minus_one_rho() {
    // m = 4, sigma^2 = 2, rho = 0.3: Var(mean) = 2 * (1 + 3 * 0.3) / 4 = 0.95.
    let spec = spec_of(declaration(
        even(),
        vec![10],
        repeated(4, 2.0, Some(compound(0.3))),
        cluster_robust(UnitSummary::Mean),
    ));
    close(spec.unit_summary_variance(), 0.95, 1e-12, "variance of the unit mean");
    close(spec.design_effect(), 1.9, 1e-12, "design effect");
    close(spec.marginal_variance(), 2.0, 0.0, "marginal variance");
    // GLS equals the mean under compound symmetry.
    let gls = spec_of(declaration(
        even(),
        vec![10],
        repeated(4, 2.0, Some(compound(0.3))),
        cluster_robust(UnitSummary::Gls),
    ));
    close(gls.unit_summary_variance(), 0.95, 1e-12, "GLS variance under compound symmetry");
    // A negative correlation inside the positive-definite range deflates it:
    // m = 2, rho = -0.5 gives 1 + 1 * (-0.5) = 0.5.
    let negative = spec_of(declaration(
        even(),
        vec![10],
        repeated(2, 1.0, Some(compound(-0.5))),
        cluster_robust(UnitSummary::Mean),
    ));
    close(negative.design_effect(), 0.5, 1e-12, "negative correlation design effect");
    // Independence declared as zero: no inflation.
    let independent = spec_of(declaration(
        even(),
        vec![10],
        repeated(5, 3.0, Some(compound(0.0))),
        cluster_robust(UnitSummary::Mean),
    ));
    close(independent.design_effect(), 1.0, 1e-12, "independent design effect");
    close(independent.unit_summary_variance(), 0.6, 1e-12, "sigma^2 / m");
}

#[test]
fn autoregressive_correlation_matches_the_hand_summed_matrix_and_gls_closed_form() {
    // m = 3, rho = 0.5: sum of the correlation matrix = 3 + 2 (2 * 0.5 + 0.25) = 5.5,
    // Var(mean) = 5.5 / 9.
    let mean = spec_of(declaration(
        even(),
        vec![10],
        repeated(3, 1.0, Some(autoregressive(0.5))),
        cluster_robust(UnitSummary::Mean),
    ));
    close(mean.unit_summary_variance(), 5.5 / 9.0, 1e-12, "AR(1) mean variance");
    // GLS: sigma^2 (1 + rho) / (2 + (m - 2) (1 - rho)) = 1.5 / 2.5 = 0.6.
    let gls = spec_of(declaration(
        even(),
        vec![10],
        repeated(3, 1.0, Some(autoregressive(0.5))),
        cluster_robust(UnitSummary::Gls),
    ));
    close(gls.unit_summary_variance(), 0.6, 1e-12, "AR(1) GLS variance");
    assert!(gls.unit_summary_variance() < mean.unit_summary_variance());
    // m = 2: GLS weights are equal, so GLS equals the mean: (1 + rho) / 2.
    let two = spec_of(declaration(
        even(),
        vec![10],
        repeated(2, 1.0, Some(autoregressive(0.4))),
        cluster_robust(UnitSummary::Gls),
    ));
    close(two.unit_summary_variance(), 0.7, 1e-12, "AR(1) m = 2");
}

#[test]
fn a_cluster_mean_has_variance_between_plus_within_over_members() {
    // tau^2 = 0.5, sigma^2 = 1.5, m = 5: Var = 0.5 + 1.5 / 5 = 0.8; ICC = 0.25;
    // design effect = 1 + 4 * 0.25 = 2 = 5 * 0.8 / 2.
    let spec = spec_of(declaration(
        even(),
        vec![10],
        OutcomeStructure::Hierarchical {
            members: 5,
            between_variance: Some(0.5),
            within_variance: Some(1.5),
        },
        cluster_robust(UnitSummary::Mean),
    ));
    close(spec.unit_summary_variance(), 0.8, 1e-12, "cluster mean variance");
    close(spec.marginal_variance(), 2.0, 1e-12, "marginal variance");
    close(spec.design_effect(), 2.0, 1e-12, "cluster design effect");
    assert_eq!(spec.outcome_name(), "hierarchical_clustered");
}

// -- Exact enumerated law of a fixed allocation ----------------------------------------------------

fn fixed_spec(units: u32, m: u32, rho: f64, summary: UnitSummary) -> RepeatedMeasuresSpec {
    spec_of(declaration(
        even(),
        vec![units],
        repeated(m, 1.0, Some(compound(rho))),
        cluster_robust(summary),
    ))
}

fn mass(law: &GaussianMixtureLaw) -> f64 {
    law.defined_mass() + law.undefined_mass
}

#[test]
fn the_fixed_allocation_contrast_law_matches_the_enumerated_binomial_mixture() {
    // m = 2, rho = 0.5, sigma^2 = 1: v = 0.75; delta = 0.4.
    let v = 0.75;
    let means = [0.0, 0.4];
    // n = 2: k = 1 with probability 2 * 1/4 = 1/2; variance v (1 + 1) = 1.5.
    let n2 = exact_fixed_contrast_law(&fixed_spec(2, 2, 0.5, UnitSummary::Mean), &means).unwrap();
    assert_eq!(n2.components.len(), 1);
    close(n2.components[0].weight, 0.5, 1e-12, "n=2 mass");
    close(n2.components[0].variance, 1.5, 1e-12, "n=2 variance");
    close(n2.undefined_mass, 0.5, 1e-12, "n=2 undefined mass");
    close(n2.mean(), 0.4, 1e-12, "n=2 mean");
    close(n2.variance(), 1.5, 1e-12, "n=2 variance of the law");
    close(n2.cdf(0.4), 0.5, 1e-9, "median of a centred normal");
    close(mass(&n2), 1.0, 1e-12, "total mass");
    // n = 3: k = 1 and k = 2 each 3/8; variance v (1 + 1/2) = 1.125; undefined 2/8.
    let n3 = exact_fixed_contrast_law(&fixed_spec(3, 2, 0.5, UnitSummary::Mean), &means).unwrap();
    assert_eq!(n3.components.len(), 2);
    for component in &n3.components {
        close(component.weight, 3.0 / 8.0, 1e-12, "n=3 mass");
        close(component.variance, v * 1.5, 1e-12, "n=3 variance");
    }
    close(n3.undefined_mass, 0.25, 1e-12, "n=3 undefined mass");
    // n = 4: k = 1, 3 with probability 4/16 and variance 4v/3; k = 2 with 6/16 and v.
    // Variance given defined = (2 * (4/16) * (4/3) + 6/16) / (14/16) v = 25/21 v.
    let n4 = exact_fixed_contrast_law(&fixed_spec(4, 2, 0.5, UnitSummary::Mean), &means).unwrap();
    assert_eq!(n4.components.len(), 3);
    close(n4.components[0].weight, 4.0 / 16.0, 1e-12, "k=1 mass");
    close(n4.components[1].weight, 6.0 / 16.0, 1e-12, "k=2 mass");
    close(n4.components[1].variance, v, 1e-12, "k=2 variance");
    close(n4.undefined_mass, 2.0 / 16.0, 1e-12, "n=4 undefined mass");
    close(n4.variance(), 25.0 / 21.0 * v, 1e-12, "n=4 variance of the law");
    // The variance is proportional to the design effect: rho = 0 against rho = 0.5.
    let independent =
        exact_fixed_contrast_law(&fixed_spec(4, 2, 0.0, UnitSummary::Mean), &means).unwrap();
    close(n4.variance() / independent.variance(), 1.5, 1e-12, "ratio is 1 + (m - 1) rho");
    // Unequal allocation: p = 0.25 plus; n = 2: k = 1 with 2 * 0.25 * 0.75 = 0.375.
    let skewed = spec_of(declaration(
        UnitAssignment::Fixed { probabilities: vec![0.75, 0.25] },
        vec![2],
        repeated(2, 1.0, Some(compound(0.5))),
        cluster_robust(UnitSummary::Mean),
    ));
    let law = exact_fixed_contrast_law(&skewed, &means).unwrap();
    close(law.components[0].weight, 0.375, 1e-12, "skewed allocation mass");
}

#[test]
fn only_revealed_units_enter_the_exact_law() {
    // Four single-unit batches with a lag of two: two units are revealed at
    // the final analysis, so the law is the n = 2 mixture.
    let mut d = declaration(
        even(),
        vec![1, 1, 1, 1],
        repeated(2, 1.0, Some(compound(0.5))),
        cluster_robust(UnitSummary::Mean),
    );
    d.lag_looks = 2;
    let spec = spec_of(d);
    assert_eq!(spec.revealed_units(), 2);
    let law = exact_fixed_contrast_law(&spec, &[0.0, 0.0]).unwrap();
    close(law.undefined_mass, 0.5, 1e-12, "two revealed units");
    // A trailing follow-up batch with no enrolment reveals the pending units.
    let mut follow_up = declaration(
        even(),
        vec![1, 1, 1, 1, 0, 0],
        repeated(2, 1.0, Some(compound(0.5))),
        cluster_robust(UnitSummary::Mean),
    );
    follow_up.lag_looks = 2;
    assert_eq!(spec_of(follow_up).revealed_units(), 4);
}

// -- Simulation against the closed forms -----------------------------------------------------------

#[test]
fn simulated_cluster_robust_contrast_matches_the_exact_mixture_under_compound_symmetry() {
    // n = 4, m = 2, rho = 0.5: exact undefined mass 1/8, conditional mean 0.4,
    // conditional variance 25/21 * 0.75.
    let spec = fixed_spec(4, 2, 0.5, UnitSummary::Mean);
    let means = [0.0, 0.4];
    let exact = exact_fixed_contrast_law(&spec, &means).unwrap();
    let law = simulate_repeated_measures(&spec, &means, &request(20_000, 11), &never).unwrap();
    let summary = law.summarize(0.4, 1.96).unwrap();
    close(summary.undefined_fraction, exact.undefined_mass, 0.008, "undefined fraction");
    assert!(
        summary.bias.abs() <= 5.0 * summary.monte_carlo_standard_error,
        "mean estimate is unbiased within 5 Monte Carlo standard errors: bias {} mcse {}",
        summary.bias,
        summary.monte_carlo_standard_error
    );
    let expected = exact.variance();
    assert!(
        (summary.variance / expected - 1.0).abs() < 0.01,
        "simulated variance {} against exact {expected}",
        summary.variance
    );
}

#[test]
fn simulated_variance_scales_with_the_design_effect() {
    // n = 20 units, m = 4: rho = 0.8 inflates the contrast variance by
    // (1 + 3 * 0.8) / 1 = 3.4 against independence, exactly and in simulation.
    let dependent = fixed_spec(20, 4, 0.8, UnitSummary::Mean);
    let independent = fixed_spec(20, 4, 0.0, UnitSummary::Mean);
    let means = [0.0, 0.0];
    let exact_dependent = exact_fixed_contrast_law(&dependent, &means).unwrap();
    let exact_independent = exact_fixed_contrast_law(&independent, &means).unwrap();
    close(
        exact_dependent.variance() / exact_independent.variance(),
        3.4,
        1e-12,
        "exact variance ratio is the design effect",
    );
    for (spec, exact) in [(&dependent, &exact_dependent), (&independent, &exact_independent)] {
        let law = simulate_repeated_measures(spec, &means, &request(12_000, 5), &never).unwrap();
        let summary = law.summarize(0.0, 1.96).unwrap();
        assert!(
            (summary.variance / exact.variance() - 1.0).abs() < 0.01,
            "simulated variance {} against exact {}",
            summary.variance,
            exact.variance()
        );
    }
}

#[test]
fn simulated_gls_summary_under_ar1_matches_the_closed_form_variance() {
    // m = 3, rho = 0.5: v = 0.6 (GLS) against 5.5 / 9 (mean); n = 6.
    let spec = spec_of(declaration(
        even(),
        vec![6],
        repeated(3, 1.0, Some(autoregressive(0.5))),
        cluster_robust(UnitSummary::Gls),
    ));
    let means = [0.0, 0.3];
    let exact = exact_fixed_contrast_law(&spec, &means).unwrap();
    close(
        exact.components.iter().map(|c| c.variance).fold(f64::MAX, f64::min),
        0.6 * (1.0 / 3.0 + 1.0 / 3.0),
        1e-12,
        "balanced allocation variance 0.6 * (1/3 + 1/3)",
    );
    let law = simulate_repeated_measures(&spec, &means, &request(15_000, 23), &never).unwrap();
    let summary = law.summarize(0.3, 1.96).unwrap();
    assert!(summary.bias.abs() <= 5.0 * summary.monte_carlo_standard_error);
    assert!(
        (summary.variance / exact.variance() - 1.0).abs() < 0.03,
        "simulated variance {} against exact {}",
        summary.variance,
        exact.variance()
    );
}

#[test]
fn a_clustered_outcome_simulates_to_the_random_effect_variance() {
    // tau^2 = 0.5, sigma^2 = 1.5, m = 5: v = 0.8; n = 12.
    let spec = spec_of(declaration(
        even(),
        vec![12],
        OutcomeStructure::Hierarchical {
            members: 5,
            between_variance: Some(0.5),
            within_variance: Some(1.5),
        },
        cluster_robust(UnitSummary::Mean),
    ));
    let means = [1.0, 1.0];
    let exact = exact_fixed_contrast_law(&spec, &means).unwrap();
    let law = simulate_repeated_measures(&spec, &means, &request(12_000, 31), &never).unwrap();
    let summary = law.summarize(0.0, 1.96).unwrap();
    assert!(summary.bias.abs() <= 5.0 * summary.monte_carlo_standard_error);
    assert!(
        (summary.variance / exact.variance() - 1.0).abs() < 0.01,
        "simulated variance {} against exact {}",
        summary.variance,
        exact.variance()
    );
}

#[test]
fn the_weighted_estimate_is_unbiased_under_epsilon_greedy_assignment() {
    // 40 units enrolled one at a time under epsilon-greedy over unit means;
    // the augmented score is a martingale difference under any history-
    // dependent policy, so the uniform weighted contrast is unbiased.
    let spec = spec_of(declaration(
        greedy(0.4),
        vec![1; 40],
        repeated(3, 1.0, Some(compound(0.5))),
        weighted(),
    ));
    assert_eq!(spec.family_name(), "epsilon_greedy");
    assert_eq!(
        spec.cell().id(),
        "epsilon_greedy/repeated_measures/final_analysis_only/no_stopping"
    );
    assert_eq!(spec.cell().paradigm, "frequentist_adaptive_weighting");
    let means = [0.0, 0.5];
    let law = simulate_repeated_measures(&spec, &means, &request(3_000, 41), &never).unwrap();
    let summary = law.summarize(0.5, 1.96).unwrap();
    assert!(summary.defined >= 2_990, "estimates are defined with 40 analysed units");
    assert!(
        summary.bias.abs() <= 5.0 * summary.monte_carlo_standard_error,
        "bias {} against 5 mcse {}",
        summary.bias,
        summary.monte_carlo_standard_error
    );
    for replicate in &law.replicates {
        assert_eq!(replicate.enrolled, 40);
        assert_eq!(replicate.analysed_units, 40);
    }
    // The policy moved toward the better arm: more units were assigned to it.
    let to_better: u32 = law.replicates.iter().map(|r| r.units_per_arm[1]).sum();
    let to_worse: u32 = law.replicates.iter().map(|r| r.units_per_arm[0]).sum();
    assert!(to_better > to_worse, "greedy assignment favours the better arm");
}

#[test]
fn the_derivation_reports_exact_forms_and_monte_carlo_error_separately() {
    let spec = fixed_spec(6, 3, 0.3, UnitSummary::Mean);
    let means = [0.0, 0.2];
    let derived =
        derive_repeated_measures(&spec, &means, &request(2_000, 7), 1.96, &never).unwrap();
    assert_eq!(derived.calibration, DERIVED_UNMEASURED);
    assert_eq!(derived.paradigm, UncertaintyParadigm::FrequentistFixedSample);
    assert_eq!(derived.route, "cluster_robust_unit_summaries");
    close(derived.design_effect, 1.0 + 2.0 * 0.3, 1e-12, "design effect");
    close(derived.unit_summary_variance, 1.6 / 3.0, 1e-12, "unit summary variance");
    close(derived.contrast_value, 0.2, 0.0, "contrast");
    assert!(derived.exact_law.is_some());
    assert!(derived.simulated.monte_carlo_standard_error > 0.0);
    assert_eq!(derived.replications, 2_000);
    // A weighted analysis has no closed-form contrast law.
    let adaptive = spec_of(declaration(
        greedy(0.5),
        vec![1; 20],
        repeated(3, 1.0, Some(compound(0.3))),
        weighted(),
    ));
    let derived =
        derive_repeated_measures(&adaptive, &means, &request(300, 7), 1.96, &never).unwrap();
    assert!(derived.exact_law.is_none());
    assert_eq!(derived.route, "adaptively_weighted_unit_scores");
    assert_eq!(derived.paradigm, UncertaintyParadigm::FrequentistAdaptiveWeighting);
}

// -- Determinism, identity of streams --------------------------------------------------------------

#[test]
fn a_replicate_does_not_depend_on_how_many_others_are_run() {
    let spec = spec_of(declaration(
        greedy(0.3),
        vec![1; 12],
        repeated(3, 1.0, Some(compound(0.4))),
        weighted(),
    ));
    let means = [0.0, 0.5];
    let short = simulate_repeated_measures(&spec, &means, &request(25, 99), &never).unwrap();
    let long = simulate_repeated_measures(&spec, &means, &request(60, 99), &never).unwrap();
    assert_eq!(short.replicates[..], long.replicates[..25]);
}

#[test]
fn a_changed_correlation_at_unchanged_shape_changes_the_simulated_law() {
    let mk = |rho| {
        spec_of(declaration(
            greedy(0.3),
            vec![1; 12],
            repeated(3, 1.0, Some(compound(rho))),
            weighted(),
        ))
    };
    let means = [0.0, 0.5];
    let a = simulate_repeated_measures(&mk(0.2), &means, &request(20, 3), &never).unwrap();
    let b = simulate_repeated_measures(&mk(0.7), &means, &request(20, 3), &never).unwrap();
    assert_ne!(a.replicates, b.replicates);
}

#[test]
fn cancellation_discards_the_simulation() {
    let spec = fixed_spec(4, 2, 0.5, UnitSummary::Mean);
    let outcome = simulate_repeated_measures(&spec, &[0.0, 0.0], &request(10, 1), &|| true);
    assert_eq!(outcome, Err(Error::Cancelled));
}

// -- Cells -----------------------------------------------------------------------------------------

#[test]
fn matrix_names_follow_the_declared_structure_and_batches() {
    let sequential = spec_of(declaration(
        greedy(0.2),
        vec![1; 10],
        repeated(3, 1.0, Some(compound(0.2))),
        weighted(),
    ));
    assert_eq!(sequential.family_name(), "epsilon_greedy");
    let batched = spec_of(declaration(
        greedy(0.2),
        vec![5, 5],
        repeated(3, 1.0, Some(compound(0.2))),
        weighted(),
    ));
    assert_eq!(batched.family_name(), "batched_adaptive");
    let clustered = spec_of(declaration(
        greedy(0.2),
        vec![1; 10],
        OutcomeStructure::Hierarchical {
            members: 4,
            between_variance: Some(1.0),
            within_variance: Some(1.0),
        },
        weighted(),
    ));
    assert_eq!(
        clustered.cell().id(),
        "epsilon_greedy/hierarchical_clustered/final_analysis_only/no_stopping"
    );
    assert_eq!(fixed_spec(10, 3, 0.2, UnitSummary::Mean).cell().family, "fixed_randomization");
}

// -- Negative witnesses ----------------------------------------------------------------------------

#[test]
fn a_missing_within_unit_correlation_is_its_own_refusal() {
    let gaps = gaps_of(declaration(
        even(),
        vec![10],
        repeated(4, 1.0, None),
        cluster_robust(UnitSummary::Mean),
    ));
    assert_eq!(gaps, vec![ContractGap::WithinUnitCorrelationMissing]);
    assert_eq!(gaps[0].detail(), "adaptive_inference.within_unit_correlation_missing");
    assert_eq!(
        gaps[0].reason_code(),
        antecedent_core::reason_code!("adaptive_contract_incomplete")
    );
    assert!(gaps[0].remedy().contains("within-unit correlation"));
}

#[test]
fn a_missing_variance_component_of_a_cluster_is_its_own_refusal() {
    let cluster = |between, within| OutcomeStructure::Hierarchical {
        members: 4,
        between_variance: between,
        within_variance: within,
    };
    let only_between_missing = gaps_of(declaration(
        even(),
        vec![10],
        cluster(None, Some(1.0)),
        cluster_robust(UnitSummary::Mean),
    ));
    assert_eq!(only_between_missing, vec![ContractGap::BetweenUnitVarianceMissing]);
    assert_eq!(
        only_between_missing[0].detail(),
        "adaptive_inference.between_unit_variance_missing"
    );
    let both = gaps_of(declaration(
        even(),
        vec![10],
        cluster(None, None),
        cluster_robust(UnitSummary::Mean),
    ));
    assert_eq!(
        both,
        vec![ContractGap::BetweenUnitVarianceMissing, ContractGap::WithinUnitVarianceMissing]
    );
}

#[test]
fn every_missing_element_gives_its_own_gap() {
    let mut d = declaration(
        even(),
        vec![10],
        repeated(4, 1.0, Some(compound(0.2))),
        cluster_robust(UnitSummary::Mean),
    );
    d.assignment = None;
    d.enrolment = None;
    d.structure = None;
    d.analysis = None;
    d.stopping = None;
    let gaps = gaps_of(d);
    assert_eq!(
        gaps,
        vec![
            ContractGap::AssignmentPolicyMissing,
            ContractGap::EnrolmentScheduleMissing,
            ContractGap::OutcomeStructureMissing,
            ContractGap::FinalAnalysisMissing,
            ContractGap::StoppingRuleMissing,
        ]
    );
    let details: Vec<&str> = gaps.iter().map(|g| g.detail()).collect();
    assert_eq!(
        details,
        vec![
            "adaptive_inference.policy_history_missing",
            "adaptive_inference.interim_analyses_missing",
            "adaptive_inference.outcome_structure_missing",
            "adaptive_inference.final_analysis_missing",
            "adaptive_inference.stopping_rule_missing",
        ]
    );
}

#[test]
fn inconsistent_elements_are_refused_with_their_own_detail() {
    let one = |structure: OutcomeStructure| {
        gaps_of(declaration(even(), vec![10], structure, cluster_robust(UnitSummary::Mean)))
    };
    assert_eq!(
        one(repeated(1, 1.0, Some(compound(0.2)))),
        vec![ContractGap::Invalid("adaptive_inference.measurements_invalid")]
    );
    assert_eq!(
        one(repeated(33, 1.0, Some(compound(0.2)))),
        vec![ContractGap::Invalid("adaptive_inference.measurements_invalid")]
    );
    assert_eq!(
        one(repeated(4, 0.0, Some(compound(0.2)))),
        vec![ContractGap::Invalid("adaptive_inference.variance_invalid")]
    );
    // rho = 1 is singular.
    assert_eq!(
        one(repeated(4, 1.0, Some(compound(1.0)))),
        vec![ContractGap::Invalid("adaptive_inference.correlation_invalid")]
    );
    // rho (m - 1) must exceed -1: -0.5 * 3 = -1.5.
    assert_eq!(
        one(repeated(4, 1.0, Some(compound(-0.5)))),
        vec![ContractGap::Invalid("adaptive_inference.correlation_invalid")]
    );
    assert_eq!(
        one(repeated(4, 1.0, Some(autoregressive(1.0)))),
        vec![ContractGap::Invalid("adaptive_inference.correlation_invalid")]
    );
    // Stopping other than no stopping is a separate cell.
    let mut stopped = declaration(
        even(),
        vec![10],
        repeated(4, 1.0, Some(compound(0.2))),
        cluster_robust(UnitSummary::Mean),
    );
    stopped.stopping = Some(StoppingRule::WaldBoundaries {
        critical: vec![2.0],
        sidedness: Sidedness::TwoSidedSymmetric,
    });
    assert_eq!(
        gaps_of(stopped),
        vec![ContractGap::Invalid("adaptive_inference.repeated_measures_stopping_not_derived")]
    );
    // The follow-up delay is bounded in batches.
    let lagged = |lag_looks: usize| {
        let mut d = declaration(
            even(),
            vec![10],
            repeated(4, 1.0, Some(compound(0.2))),
            cluster_robust(UnitSummary::Mean),
        );
        d.lag_looks = lag_looks;
        d
    };
    assert!(RepeatedMeasuresSpec::new(lagged(MAX_UNIT_LAG)).is_ok());
    assert_eq!(
        gaps_of(lagged(MAX_UNIT_LAG + 1)),
        vec![ContractGap::Invalid("adaptive_inference.outcome_availability_invalid")]
    );
    // A contrast of an arm with itself.
    let mut same = declaration(
        even(),
        vec![10],
        repeated(4, 1.0, Some(compound(0.2))),
        cluster_robust(UnitSummary::Mean),
    );
    same.analysis =
        Some(FinalAnalysis { estimator: cluster_robust(UnitSummary::Mean), plus: 1, minus: 1 });
    assert_eq!(gaps_of(same), vec![ContractGap::Invalid("adaptive_inference.analysis_invalid")]);
}

#[test]
fn a_fixed_sample_analysis_after_adaptive_assignment_is_refused() {
    let spec = spec_of(declaration(
        greedy(0.3),
        vec![1; 20],
        repeated(3, 1.0, Some(compound(0.3))),
        cluster_robust(UnitSummary::Mean),
    ));
    assert_eq!(
        premise_detail(admit_repeated_measures(&spec, UncertaintyParadigm::FrequentistFixedSample)),
        "adaptive_inference.fixed_sample_design_adaptive"
    );
    // The adaptive-weighting paradigm with a fixed-sample contrast is the same refusal.
    assert_eq!(
        premise_detail(admit_repeated_measures(
            &spec,
            UncertaintyParadigm::FrequentistAdaptiveWeighting
        )),
        "adaptive_inference.fixed_sample_design_adaptive"
    );
    // Neither the simulation nor the derivation runs.
    assert_eq!(
        premise_detail(simulate_repeated_measures(&spec, &[0.0, 0.2], &request(10, 1), &never)),
        "adaptive_inference.fixed_sample_design_adaptive"
    );
    assert_eq!(
        premise_detail(derive_repeated_measures(&spec, &[0.0, 0.2], &request(10, 1), 1.96, &never)),
        "adaptive_inference.fixed_sample_design_adaptive"
    );
}

#[test]
fn the_independence_assumption_is_refused_for_dependent_measurements() {
    let dependent = spec_of(declaration(
        even(),
        vec![10],
        repeated(4, 1.0, Some(compound(0.3))),
        AnalysisEstimator::PooledIndependent,
    ));
    assert_eq!(
        premise_detail(admit_repeated_measures(
            &dependent,
            UncertaintyParadigm::FrequentistFixedSample
        )),
        "adaptive_inference.independence_assumed_for_dependent_measurements"
    );
    // A cluster with a random effect is dependent too.
    let clustered = spec_of(declaration(
        even(),
        vec![10],
        OutcomeStructure::Hierarchical {
            members: 4,
            between_variance: Some(0.5),
            within_variance: Some(1.0),
        },
        AnalysisEstimator::PooledIndependent,
    ));
    assert_eq!(
        premise_detail(simulate_repeated_measures(
            &clustered,
            &[0.0, 0.0],
            &request(10, 1),
            &never
        )),
        "adaptive_inference.independence_assumed_for_dependent_measurements"
    );
    // Without a declared correlation the declaration itself is refused, so an
    // iid analysis cannot be reached by omission.
    assert_eq!(
        gaps_of(declaration(
            even(),
            vec![10],
            repeated(4, 1.0, None),
            AnalysisEstimator::PooledIndependent
        )),
        vec![ContractGap::WithinUnitCorrelationMissing]
    );
    // Declared independence (rho = 0) is admitted and its variance is the
    // independent closed form: n * sigma^2 / m per arm.
    let independent = spec_of(declaration(
        even(),
        vec![10],
        repeated(4, 1.0, Some(compound(0.0))),
        AnalysisEstimator::PooledIndependent,
    ));
    admit_repeated_measures(&independent, UncertaintyParadigm::FrequentistFixedSample).unwrap();
    let law =
        simulate_repeated_measures(&independent, &[0.0, 0.0], &request(2_000, 17), &never).unwrap();
    let summary = law.summarize(0.0, 1.96).unwrap();
    // The pooled plug-in variance is unbiased for the exact conditional
    // variance of the contrast under independence.
    let exact = exact_fixed_contrast_law(&independent, &[0.0, 0.0]).unwrap();
    assert!(
        (summary.mean_reported_variance / exact.variance() - 1.0).abs() < 0.02,
        "reported {} against exact {}",
        summary.mean_reported_variance,
        exact.variance()
    );
}

#[test]
fn other_paradigms_are_separate_cells_and_not_derived() {
    let spec = spec_of(declaration(
        greedy(0.3),
        vec![1; 20],
        repeated(3, 1.0, Some(compound(0.3))),
        weighted(),
    ));
    for paradigm in [
        UncertaintyParadigm::BayesianPosterior,
        UncertaintyParadigm::AnytimeValidFrequentist,
        UncertaintyParadigm::FrequentistGroupSequential,
        UncertaintyParadigm::DecisionTheoreticStopping,
        UncertaintyParadigm::WholeProcedureInversion,
    ] {
        assert_eq!(
            premise_detail(admit_repeated_measures(&spec, paradigm)),
            "adaptive_inference.repeated_measures_paradigm_not_derived",
            "{}",
            paradigm.as_str()
        );
    }
    // The weighting analysis is not a fixed-sample analysis.
    assert_eq!(
        premise_detail(admit_repeated_measures(&spec, UncertaintyParadigm::FrequentistFixedSample)),
        "adaptive_inference.fixed_sample_design_adaptive"
    );
}

#[test]
fn a_guarantee_is_never_paired_across_cells() {
    let spec = spec_of(declaration(
        greedy(0.3),
        vec![1; 20],
        repeated(3, 1.0, Some(compound(0.3))),
        weighted(),
    ));
    let cell = spec.cell();
    assert_eq!(cell.id(), "epsilon_greedy/repeated_measures/final_analysis_only/no_stopping");
    // The same cell: accepted.
    cell.transfer_to(&spec.cell()).unwrap();
    // The clustered outcome of the same family is a different cell.
    let clustered = spec_of(declaration(
        greedy(0.3),
        vec![1; 20],
        OutcomeStructure::Hierarchical {
            members: 3,
            between_variance: Some(0.5),
            within_variance: Some(1.0),
        },
        weighted(),
    ));
    assert_eq!(
        premise_detail(cell.transfer_to(&clustered.cell())),
        "adaptive_inference.guarantee_not_transferable"
    );
    // The single-outcome cell of the same family and paradigm is a different cell.
    assert_eq!(
        premise_detail(cell.transfer_to_named(
            "epsilon_greedy/single_outcome/final_analysis_only/no_stopping",
            "frequentist_adaptive_weighting"
        )),
        "adaptive_inference.guarantee_not_transferable"
    );
    // The same axes under another paradigm is a different statement.
    assert_eq!(
        premise_detail(cell.transfer_to_named(
            "epsilon_greedy/repeated_measures/final_analysis_only/no_stopping",
            "anytime_valid_frequentist"
        )),
        "adaptive_inference.guarantee_not_transferable"
    );
    // A fixed-allocation guarantee does not carry to the adaptive family.
    let fixed = fixed_spec(10, 3, 0.3, UnitSummary::Mean);
    assert_eq!(
        premise_detail(fixed.cell().transfer_to(&cell)),
        "adaptive_inference.guarantee_not_transferable"
    );
}

#[test]
fn an_adaptive_or_weighted_procedure_has_no_closed_form_law() {
    let spec = spec_of(declaration(
        greedy(0.3),
        vec![1; 20],
        repeated(3, 1.0, Some(compound(0.3))),
        weighted(),
    ));
    assert_eq!(
        premise_detail(exact_fixed_contrast_law(&spec, &[0.0, 0.0])),
        "adaptive_inference.exact_law_unavailable"
    );
    let fixed = fixed_spec(10, 3, 0.3, UnitSummary::Mean);
    assert_eq!(exact_fixed_contrast_law(&fixed, &[0.0]), Err(Error::Shape("one mean per arm")));
}

#[test]
fn a_request_outside_bounds_is_refused() {
    let spec = fixed_spec(10, 3, 0.3, UnitSummary::Mean);
    assert!(matches!(
        simulate_repeated_measures(&spec, &[0.0, 0.0], &request(0, 1), &never),
        Err(Error::InvalidParameter(_))
    ));
    assert!(matches!(
        simulate_repeated_measures(&spec, &[0.0, 0.0], &request(1_000_001, 1), &never),
        Err(Error::Capacity(_))
    ));
    assert!(matches!(
        simulate_repeated_measures(&spec, &[0.0, f64::NAN], &request(10, 1), &never),
        Err(Error::InvalidParameter(_))
    ));
}

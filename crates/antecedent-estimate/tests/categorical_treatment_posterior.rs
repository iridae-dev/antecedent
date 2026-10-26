//! Categorical-treatment posterior against an independent derivation.
//!
//! Reference values come from an augmented-data least-squares route: the Normal-Inverse-Gamma
//! prior is written as pseudo-observations, the posterior mean is the least-squares solution of
//! the stacked system, `b_n = b0 + RSS_stacked / 2`, and the contrast quantiles are those of
//! `scipy.stats.t`. Dataset: `a = [1, 3]`, `b = [4, 6, 5]`, `c = [9, 7, 11, 9]` (rows
//! interleaved), reference `a`, level precision 0.5, intercept precision 0.01, `a0 = 2`,
//! `b0 = 1`. Then `a_n = 6.5` and `b_n = 16.515140670594`.

use antecedent_estimate::categorical_treatment::{
    CategoricalTreatmentInput, CategoricalTreatmentSpec, LevelScale, MonotonicityDirection,
};
use antecedent_estimate::categorical_treatment_posterior::{
    CATEGORICAL_POSTERIOR_ESTIMAND, CategoricalPosteriorPrior, posterior_categorical_treatment,
};
use antecedent_estimate::vector_treatment::VectorCovariance;

#[path = "common/checks.rs"]
mod checks;

fn close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= 1e-8, "{what}: {actual} vs {expected}");
}

fn input() -> CategoricalTreatmentInput<'static> {
    let rows = [
        ("c", 9.0),
        ("a", 1.0),
        ("b", 4.0),
        ("c", 7.0),
        ("b", 6.0),
        ("a", 3.0),
        ("c", 11.0),
        ("b", 5.0),
        ("c", 9.0),
    ];
    CategoricalTreatmentInput {
        outcome: rows.iter().map(|r| r.1).collect(),
        row_snapshot: "snap".into(),
        adjustment: vec![],
        levels: rows.iter().map(|r| r.0.to_string()).collect(),
    }
}

fn spec(scale: LevelScale) -> CategoricalTreatmentSpec {
    CategoricalTreatmentSpec {
        declared_levels: vec!["a".into(), "b".into(), "c".into()],
        scale,
        reference: "a".into(),
        min_level_rows: 2,
        pairwise: vec![("b".into(), "c".into())],
        monotonicity: None,
        covariance: VectorCovariance::ModelBased,
    }
}

fn prior() -> CategoricalPosteriorPrior {
    CategoricalPosteriorPrior {
        level_precision: 0.5,
        nuisance_precision: 0.01,
        variance_shape: 2.0,
        variance_scale: 1.0,
    }
}

#[test]
fn level_and_pair_posteriors_match_augmented_least_squares_and_student_t() {
    let posterior =
        posterior_categorical_treatment(&input(), &spec(LevelScale::Unordered), &prior(), 0.9)
            .unwrap();
    assert_eq!(posterior.estimand, CATEGORICAL_POSTERIOR_ESTIMAND);
    let (shape, scale) = posterior.variance_shape_and_scale;
    close(shape, 6.5, "a_n");
    close(scale, 16.515_140_670_594, "b_n");
    let b = &posterior.level_contrasts[0];
    close(b.mean, 1.270_164_620_382, "b - a mean");
    close(b.posterior.scale, 1.171_931_378_603, "b - a scale");
    close(b.posterior.degrees_of_freedom, 13.0, "df");
    close(b.interval[0], -0.805_247_795_790, "b - a lo");
    close(b.interval[1], 3.345_577_036_554, "b - a hi");
    close(b.probability_positive, 0.850_931_905_979, "P(b > a)");
    let c = &posterior.level_contrasts[1];
    close(c.mean, 4.872_763_310_026, "c - a mean");
    close(c.interval[0], 2.884_143_902_253, "c - a lo");
    close(c.interval[1], 6.861_382_717_798, "c - a hi");
    close(c.probability_positive, 0.999_598_689_489, "P(c > a)");
    // The pair uses the off-diagonal of the posterior covariance, not the two marginals.
    let pair = &posterior.pairwise[0];
    close(pair.mean, 3.602_598_689_644, "c - b mean");
    close(pair.posterior.scale, 1.136_419_210_808, "c - b scale");
    close(pair.interval[0], 1.590_075_957_383, "c - b lo");
    close(pair.interval[1], 5.615_121_421_905, "c - b hi");
    close(pair.probability_positive, 0.996_310_007_347, "P(c > b)");
}

#[test]
fn monotonicity_statement_is_bonferroni_and_frechet_bounds() {
    let mut ordered = spec(LevelScale::Ordered);
    ordered.monotonicity = Some(MonotonicityDirection::NonDecreasing);
    let posterior = posterior_categorical_treatment(&input(), &ordered, &prior(), 0.9).unwrap();
    let mono = posterior.monotonicity.unwrap();
    close(mono.step_probabilities[0], 0.850_931_905_979, "P(step ab up)");
    close(mono.step_probabilities[1], 0.996_310_007_347, "P(step bc up)");
    close(mono.lower_bound, 0.847_241_913_326, "Bonferroni lower bound");
    close(mono.upper_bound, 0.850_931_905_979, "Frechet upper bound");
    assert!(mono.lower_bound <= mono.upper_bound);
    // The opposite direction is the complement per step.
    ordered.monotonicity = Some(MonotonicityDirection::NonIncreasing);
    let down = posterior_categorical_treatment(&input(), &ordered, &prior(), 0.9).unwrap();
    let down = down.monotonicity.unwrap();
    close(down.step_probabilities[0], 1.0 - 0.850_931_905_979, "P(step ab down)");
}

#[test]
fn prior_credible_level_and_design_refusals_are_typed() {
    let bad_prior = CategoricalPosteriorPrior { level_precision: 0.0, ..prior() };
    let error =
        posterior_categorical_treatment(&input(), &spec(LevelScale::Unordered), &bad_prior, 0.9)
            .unwrap_err();
    let (code, message) = checks::coded_refusal(error);
    assert_eq!(code, "invalid_argument");
    assert!(message.contains("gaussian_linear_posterior.invalid_prior"), "{message}");

    let error =
        posterior_categorical_treatment(&input(), &spec(LevelScale::Unordered), &prior(), 1.0)
            .unwrap_err();
    let (_, message) = checks::coded_refusal(error);
    assert!(message.contains("categorical_treatment_posterior.invalid_level"), "{message}");

    let mut sparse = spec(LevelScale::Unordered);
    sparse.min_level_rows = 3;
    let error = posterior_categorical_treatment(&input(), &sparse, &prior(), 0.9).unwrap_err();
    let (code, message) = checks::coded_refusal(error);
    assert_eq!(code, "arm_not_populated");
    assert!(message.contains("categorical_treatment.sparse_level"), "{message}");
}

#[test]
fn weak_prior_posterior_mean_approaches_the_frequentist_group_difference() {
    let weak = CategoricalPosteriorPrior {
        level_precision: 1e-8,
        nuisance_precision: 1e-8,
        variance_shape: 1.0,
        variance_scale: 1.0,
    };
    let posterior =
        posterior_categorical_treatment(&input(), &spec(LevelScale::Unordered), &weak, 0.95)
            .unwrap();
    // Group means are 2, 5, 9 (dataset of the frequentist fixture).
    assert!((posterior.level_contrasts[0].mean - 3.0).abs() < 1e-6);
    assert!((posterior.level_contrasts[1].mean - 7.0).abs() < 1e-6);
    assert!((posterior.pairwise[0].mean - 4.0).abs() < 1e-6);
}

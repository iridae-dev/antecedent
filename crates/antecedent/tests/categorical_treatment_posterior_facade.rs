//! The posterior section of the categorical-treatment artifact through the public facade: the
//! Student-t contrasts against a numpy/scipy normal-equations reference, the declared prior and the
//! stated estimand carried with the section, the recomputing consumer that refuses an edited or
//! resealed posterior, and the typed refusals of an invalid prior or credible level.
//!
//! Reference (computed outside the library): `y | beta, s2 ~ N(X beta, s2 I)`, prior precision
//! `diag(0.5, 1, 1)` on `(intercept, level coefficients)`, `s2 ~ InvGamma(2, 1)`; posterior
//! `Lambda_n = Lambda_0 + X'X`, `m_n = Lambda_n^-1 X'y`, `a_n = a_0 + n/2`,
//! `b_n = b_0 + (y'y - X'y . m_n)/2`, and `w'beta` is Student-t with `2 a_n` degrees of freedom,
//! location `w'm_n` and squared scale `(b_n/a_n) w' Lambda_n^-1 w`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::categorical_treatment::{
    CategoricalPosteriorDeclaration, CategoricalSpecWire, CategoricalTreatment,
    CategoricalTreatmentRequest, CovarianceKindWire, LevelScaleWire, MonotonicityDirectionWire,
};
use antecedent_io::categorical_treatment_artifact::{decode_parts, encode_parts};

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!((actual - expected).abs() <= tolerance, "{what}: {actual} vs {expected}");
}

fn names(levels: &[&str]) -> Vec<String> {
    levels.iter().map(|l| (*l).to_owned()).collect()
}

fn declaration(level_precision: f64) -> CategoricalPosteriorDeclaration {
    CategoricalPosteriorDeclaration {
        level_precision,
        nuisance_precision: 0.5,
        variance_shape: 2.0,
        variance_scale: 1.0,
        credible_level: 0.95,
    }
}

fn abc_request() -> CategoricalTreatmentRequest {
    let rows = [
        ("a", 1.0),
        ("a", 3.0),
        ("b", 4.0),
        ("b", 6.0),
        ("b", 5.0),
        ("c", 9.0),
        ("c", 7.0),
        ("c", 11.0),
        ("c", 9.0),
    ];
    CategoricalTreatmentRequest {
        outcome: rows.iter().map(|r| r.1).collect(),
        row_snapshot: "abc".into(),
        adjustment: vec![],
        levels: rows.iter().map(|r| r.0.to_owned()).collect(),
        spec: CategoricalSpecWire {
            declared_levels: names(&["a", "b", "c"]),
            scale: LevelScaleWire::Unordered,
            reference: "a".into(),
            min_level_rows: 2,
            pairwise: vec![("b".into(), "c".into())],
            monotonicity: None,
            covariance: CovarianceKindWire::ModelBased,
        },
    }
}

fn ordered_request() -> CategoricalTreatmentRequest {
    let mut outcome = vec![];
    let mut levels = vec![];
    for (label, values) in
        [("lo", [3.0, 2.0, 1.0]), ("mid", [2.0, 1.0, 0.0]), ("hi", [1.0, 0.0, -1.0])]
    {
        for value in values {
            outcome.push(value);
            levels.push(label.to_owned());
        }
    }
    CategoricalTreatmentRequest {
        outcome,
        row_snapshot: "ordered".into(),
        adjustment: vec![],
        levels,
        spec: CategoricalSpecWire {
            declared_levels: names(&["lo", "mid", "hi"]),
            scale: LevelScaleWire::Ordered,
            reference: "lo".into(),
            min_level_rows: 2,
            pairwise: vec![],
            monotonicity: Some(MonotonicityDirectionWire::NonDecreasing),
            covariance: CovarianceKindWire::ModelBased,
        },
    }
}

#[test]
fn the_posterior_contrasts_match_the_normal_equations_reference() {
    let sealed = CategoricalTreatment::evaluate(&abc_request())
        .unwrap()
        .with_posterior(&declaration(1.0))
        .unwrap();
    let posterior = sealed.posterior().expect("a posterior section");
    close(posterior.variance_shape, 6.5, 1e-12, "a_n");
    close(posterior.variance_scale, 25.182_098_765_432, 1e-9, "b_n");
    // (location, scale, 95% interval, P(> 0), standard deviation) per contrast.
    let expected = [
        (
            0.981_481_481_481,
            1.227_445_026_736,
            [-1.670_252_281_810, 3.633_215_244_773],
            0.780_848_965_647,
            1.334_373_425_230,
        ),
        (
            4.246_913_580_247,
            1.177_730_094_340,
            [1.702_582_398_662, 6.791_244_761_832],
            0.998_402_555_110,
            1.280_327_595_738,
        ),
    ];
    for (c, (location, scale, interval, positive, sd)) in
        posterior.level_contrasts.iter().zip(expected)
    {
        close(c.location, location, 1e-9, "location");
        close(c.scale, scale, 1e-9, "scale");
        close(c.degrees_of_freedom, 13.0, 1e-12, "df");
        close(c.interval[0], interval[0], 1e-8, "lower");
        close(c.interval[1], interval[1], 1e-8, "upper");
        close(c.probability_positive, positive, 1e-9, "P(> 0)");
        close(c.standard_deviation.expect("df > 2"), sd, 1e-9, "sd");
    }
    let pair = &posterior.pairwise[0];
    assert_eq!((pair.from.as_str(), pair.to.as_str()), ("b", "c"));
    close(pair.location, 3.265_432_098_765, 1e-9, "pair location");
    close(pair.scale, 1.321_274_964_096, 1e-9, "pair scale");
    close(pair.interval[0], 0.410_991_079_763, 1e-8, "pair lower");
    close(pair.probability_positive, 0.985_970_325_135, 1e-9, "pair P(> 0)");
}

#[test]
fn the_section_states_its_estimand_prior_and_standing() {
    let sealed = CategoricalTreatment::evaluate(&abc_request())
        .unwrap()
        .with_posterior(&declaration(1.0))
        .unwrap();
    let posterior = sealed.posterior().unwrap();
    assert!(posterior.estimand.starts_with("superpopulation_dummy_coefficient_contrast"));
    assert_eq!(posterior.likelihood, "gaussian_homoskedastic_dummy_regression");
    assert_eq!(posterior.calibration, "unmeasured");
    assert!(!posterior.licensed);
    assert_eq!(posterior.declaration, declaration(1.0));
    assert_eq!(sealed.report().posterior.as_ref(), Some(posterior));
    // The frequentist fit is the same with and without the section.
    let plain = CategoricalTreatment::evaluate(&abc_request()).unwrap();
    assert!(plain.posterior().is_none());
    assert_eq!(plain.fit(), sealed.fit());
}

#[test]
fn the_ordered_levels_report_monotonicity_bounds() {
    let sealed = CategoricalTreatment::evaluate(&ordered_request())
        .unwrap()
        .with_posterior(&declaration(1.0))
        .unwrap();
    let mono = sealed.posterior().unwrap().monotonicity.as_ref().expect("monotonicity");
    assert_eq!(mono.direction, "non_decreasing");
    // P(step >= 0) per adjacent step, from scipy.stats.t; Bonferroni lower and Frechet upper bound.
    close(mono.step_probabilities[0], 0.326_276_314_774, 1e-9, "first step");
    close(mono.step_probabilities[1], 0.141_277_992_926, 1e-9, "second step");
    close(mono.lower_bound, 0.0, 1e-12, "lower bound");
    close(mono.upper_bound, 0.141_277_992_926, 1e-9, "upper bound");
}

#[test]
fn an_independent_consumer_recomputes_the_section_and_refuses_edits() {
    let sealed = CategoricalTreatment::evaluate(&abc_request())
        .unwrap()
        .with_posterior(&declaration(1.0))
        .unwrap();
    let bytes = sealed.export("posterior").unwrap();
    let again = CategoricalTreatment::consume(&bytes, Some(sealed.identity())).unwrap();
    assert_eq!(again.posterior(), sealed.posterior());

    // A resealed posterior edit is caught by the recomputation.
    let (mut meta, numbers) = decode_parts(&bytes).unwrap();
    meta.posterior.as_mut().unwrap().level_contrasts[0].location += 0.5;
    let forged = encode_parts(&meta, &numbers, "forged").unwrap();
    assert_eq!(CategoricalTreatment::consume(&forged, None).err().unwrap().to_string(), "stored categorical treatment result does not replay: posterior contrasts, moments or monotonicity bounds");

    // So is a changed prior that no longer replays to the stored contrasts.
    let (mut meta, numbers) = decode_parts(&bytes).unwrap();
    meta.posterior.as_mut().unwrap().declaration.level_precision = 4.0;
    let forged = encode_parts(&meta, &numbers, "forged-prior").unwrap();
    assert_eq!(CategoricalTreatment::consume(&forged, None).err().unwrap().to_string(), "stored categorical treatment result does not replay: posterior contrasts, moments or monotonicity bounds");

    // And a claimed license.
    let (mut meta, numbers) = decode_parts(&bytes).unwrap();
    meta.posterior.as_mut().unwrap().licensed = true;
    let forged = encode_parts(&meta, &numbers, "forged-license").unwrap();
    assert_eq!(CategoricalTreatment::consume(&forged, None).err().unwrap().to_string(), "stored categorical treatment result does not replay: posterior contrasts, moments or monotonicity bounds");

    // An artifact sealed without the section stays readable and carries none.
    let plain = CategoricalTreatment::evaluate(&abc_request()).unwrap().export("plain").unwrap();
    assert!(CategoricalTreatment::consume(&plain, None).unwrap().posterior().is_none());
}

#[test]
fn an_invalid_prior_or_level_is_a_typed_refusal() {
    let sealed = CategoricalTreatment::evaluate(&abc_request()).unwrap();
    let error = sealed.clone().with_posterior(&declaration(0.0)).unwrap_err();
    let (code, detail, _) = error.refusal().expect("a coded refusal");
    assert_eq!(code, "invalid_argument");
    assert!(detail.contains("invalid_prior"), "{detail}");
    let mut level = declaration(1.0);
    level.credible_level = 1.5;
    let error = sealed.with_posterior(&level).unwrap_err();
    let (code, detail, _) = error.refusal().expect("a coded refusal");
    assert_eq!(code, "invalid_argument");
    assert!(detail.contains("invalid_level"), "{detail}");
}

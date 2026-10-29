//! Observation-conditioned queries against hand-computed and enumerated truth.
//!
//! Confounded finite model (all variables binary, levels "0" / "1"):
//!   Z ~ Bernoulli(0.5)
//!   P(S = 1 | Z = 0) = 0.2, P(S = 1 | Z = 1) = 0.8
//!   P(Y = 1 | S, Z): (0,0) 0.1, (0,1) 0.5, (1,0) 0.3, (1,1) 0.9   (keyed (S, Z))
//! so that
//!   P(S = 1) = 0.5
//!   P(Y = 1, S = 1) = 0.5*0.2*0.3 + 0.5*0.8*0.9 = 0.39  ->  P(Y=1 | S=1) = 0.78
//!   P(Y = 1, S = 0) = 0.5*0.8*0.1 + 0.5*0.2*0.5 = 0.09  ->  P(Y=1 | S=0) = 0.18
//!   P(Y = 1 | do(S=1)) = 0.5*0.3 + 0.5*0.9 = 0.6
//!   P(Y = 1 | do(S=0)) = 0.5*0.1 + 0.5*0.5 = 0.3
//!
//! Measured construct (C is the construct, M a binary indicator with
//! sensitivity 0.9 and specificity 0.8, prevalence 0.1, P(Y=1|C=0)=0.1,
//! P(Y=1|C=1)=0.7):
//!   P(C=1 | M=1) = 0.09 / (0.09 + 0.18) = 1/3,  P(Y=1 | M=1) = 0.3
//!   P(C=1 | M=0) = 0.01 / (0.01 + 0.72) = 1/73, P(Y=1 | M=0) = 7.9/73
//!   whereas P(Y=1 | C=1) = 0.7 is the value a "measured as construct" read gives.

#![allow(
    clippy::needless_range_loop,
    reason = "the enumerated oracle indexes its tables by the level of each variable"
)]

use antecedent_core::ExecutionContext;
use antecedent_model::compose::{
    ComposedFinite, ComposedModelSpec, Domain, MechanismContext, MechanismDescriptor,
    MechanismForm, MechanismOrigin, MechanismProvenance, ParentBinding, PosteriorDependence,
    TabularRow,
};
use antecedent_model::observation_query::{
    BetaBernoulli, Direction, FiniteBayesModel, ObservationQueryError, UnreachableBound, condition,
    condition_on_measured, conditioning_and_intervention, intervene,
    observations_required_binary_signal, observations_required_conjugate, required_bayes_factor,
    threshold_crossing_observations,
};

const TOL: f64 = 1e-12;

fn near(a: f64, b: f64) {
    assert!((a - b).abs() <= TOL, "{a} vs {b}");
}

fn level(levels: &[(String, f64)], name: &str) -> f64 {
    levels.iter().find(|(l, _)| l == name).map(|(_, p)| *p).unwrap()
}

fn table(
    var: &str,
    parents: &[(&str, &[&str])],
    levels: &[&str],
    rows: &[(&[&str], &[f64])],
) -> MechanismDescriptor {
    let level_names: Vec<String> = levels.iter().map(|s| (*s).to_owned()).collect();
    MechanismDescriptor {
        variable: var.into(),
        form: MechanismForm::Tabular {
            levels: level_names.clone(),
            rows: rows
                .iter()
                .map(|(config, probs)| TabularRow {
                    parent_levels: parents
                        .iter()
                        .zip(config.iter())
                        .map(|((p, _), l)| ((*p).to_owned(), (*l).to_owned()))
                        .collect(),
                    probabilities: level_names.iter().cloned().zip(probs.iter().copied()).collect(),
                })
                .collect(),
        },
        parents: parents
            .iter()
            .map(|(p, lv)| ParentBinding {
                variable: (*p).to_owned(),
                units: "category".into(),
                transform: "identity".into(),
                domain: Domain::Levels(lv.iter().map(|s| (*s).to_owned()).collect()),
            })
            .collect(),
        conditioning_set: parents.iter().map(|(p, _)| (*p).to_owned()).collect(),
        output_units: "category".into(),
        output_transform: "identity".into(),
        output_support: Domain::Levels(level_names),
        context: MechanismContext { population: "pop".into(), regime: "observational".into() },
        provenance: MechanismProvenance {
            origin: MechanismOrigin::Declared,
            artifact_id: format!("declared:{var}"),
            evidence_id: format!("assumed:{var}"),
        },
        latents: Vec::new(),
        uncertainty: None,
    }
}

fn spec_of(mechanisms: Vec<MechanismDescriptor>) -> ComposedModelSpec {
    let mut edges = Vec::new();
    for m in &mechanisms {
        for p in &m.parents {
            edges.push((p.variable.clone(), m.variable.clone()));
        }
    }
    ComposedModelSpec {
        target: MechanismContext { population: "pop".into(), regime: "observational".into() },
        mechanisms,
        edges,
        invariances: Vec::new(),
        selection_diagrams: Vec::new(),
        dependence: PosteriorDependence::Unknown,
        admit_declared_mechanisms: true,
    }
}

const B: &[&str] = &["0", "1"];

fn confounded(signal_given_confounder: [f64; 2]) -> ComposedFinite {
    let [s0, s1] = signal_given_confounder;
    ComposedFinite::compile(&spec_of(vec![
        table("Z", &[], B, &[(&[], &[0.5, 0.5])]),
        table("S", &[("Z", B)], B, &[(&["0"], &[1.0 - s0, s0]), (&["1"], &[1.0 - s1, s1])]),
        table(
            "Y",
            &[("S", B), ("Z", B)],
            B,
            &[
                (&["0", "0"], &[0.9, 0.1]),
                (&["0", "1"], &[0.5, 0.5]),
                (&["1", "0"], &[0.7, 0.3]),
                (&["1", "1"], &[0.1, 0.9]),
            ],
        ),
    ]))
    .unwrap()
}

/// Independent enumeration of the confounded model from its structural
/// equations: a nested loop over (z, s, y), not the factorization under test.
fn enumerated(s_given_z: [f64; 2], do_s: Option<usize>) -> f64 {
    let py1 = |s: usize, z: usize| match (s, z) {
        (0, 0) => 0.1,
        (0, 1) => 0.5,
        (1, 0) => 0.3,
        _ => 0.9,
    };
    let mut p_y1 = 0.0;
    let mut p_s = 0.0;
    for z in 0..2 {
        for s in 0..2 {
            let ps = match do_s {
                Some(v) => f64::from(u8::from(v == s)),
                None => {
                    if s == 1 {
                        s_given_z[z]
                    } else {
                        1.0 - s_given_z[z]
                    }
                }
            };
            let joint = 0.5 * ps;
            // Only the S = 1 slice is reported.
            if s == 1 {
                p_s += joint;
                p_y1 += joint * py1(s, z);
            }
        }
    }
    match do_s {
        Some(_) => p_y1,
        None => p_y1 / p_s,
    }
}

#[test]
fn conditioning_and_intervention_disagree_on_a_confounded_model_and_both_are_exact() {
    let ctx = ExecutionContext::for_tests(1);
    let model = confounded([0.2, 0.8]);
    let pair = conditioning_and_intervention(&model, "Y", &[("S", "1")], &ctx).unwrap();

    near(pair.conditioning.evidence_probability, 0.5);
    near(level(&pair.conditioning.levels, "1"), 0.78);
    near(level(&pair.conditioning.levels, "0"), 0.22);
    near(level(&pair.intervention, "1"), 0.6);
    near(level(&pair.intervention, "0"), 0.4);
    near(pair.max_abs_difference, 0.18);

    // The independent enumeration agrees with both values.
    near(enumerated([0.2, 0.8], None), 0.78);
    near(enumerated([0.2, 0.8], Some(1)), 0.6);

    // S = 0: 0.18 against 0.3.
    let zero = conditioning_and_intervention(&model, "Y", &[("S", "0")], &ctx).unwrap();
    near(level(&zero.conditioning.levels, "1"), 0.18);
    near(level(&zero.intervention, "1"), 0.3);
    near(zero.max_abs_difference, 0.12);
}

#[test]
fn conditioning_equals_intervention_when_the_signal_has_no_common_cause_with_the_outcome() {
    let ctx = ExecutionContext::for_tests(1);
    // S independent of Z: P(S=1 | Z) = 0.5 for both values of Z.
    let model = confounded([0.5, 0.5]);
    let pair = conditioning_and_intervention(&model, "Y", &[("S", "1")], &ctx).unwrap();
    near(level(&pair.conditioning.levels, "1"), 0.6);
    near(level(&pair.intervention, "1"), 0.6);
    near(pair.max_abs_difference, 0.0);
    near(enumerated([0.5, 0.5], None), 0.6);
}

#[test]
fn conditioning_uses_the_observational_law_and_intervention_the_truncated_factorization() {
    let ctx = ExecutionContext::for_tests(1);
    let model = confounded([0.2, 0.8]);
    let law = model.law(&[], &ctx).unwrap();
    // Conditioning on a descendant-free event leaves the marginal untouched.
    let nothing = condition(&law, "Y", &[], &ctx).unwrap();
    near(nothing.evidence_probability, 1.0);
    // P(Y=1) = 0.39 + 0.09 = 0.48
    near(level(&nothing.levels, "1"), 0.48);
    // Conditioning on both S and Z: P(Y=1 | S=1, Z=1) = 0.9 exactly.
    let both = condition(&law, "Y", &[("S", "1"), ("Z", "1")], &ctx).unwrap();
    near(both.evidence_probability, 0.4);
    near(level(&both.levels, "1"), 0.9);
    // Adjusting for Z recovers the interventional value: sum_z P(z) P(Y=1 | S=1, z).
    let z0 = condition(&law, "Y", &[("S", "1"), ("Z", "0")], &ctx).unwrap();
    near(0.5 * level(&z0.levels, "1") + 0.5 * level(&both.levels, "1"), 0.6);
    let interventional = intervene(&model, "Y", &[("S", "1")], &ctx).unwrap();
    near(level(&interventional, "1"), 0.6);
}

#[test]
fn conditioning_refuses_null_evidence_the_outcome_as_evidence_and_unknown_names() {
    let ctx = ExecutionContext::for_tests(1);
    let model = ComposedFinite::compile(&spec_of(vec![
        table("S", &[], B, &[(&[], &[1.0, 0.0])]),
        table("Y", &[("S", B)], B, &[(&["0"], &[0.5, 0.5]), (&["1"], &[0.5, 0.5])]),
    ]))
    .unwrap();
    let law = model.law(&[], &ctx).unwrap();
    assert_eq!(
        condition(&law, "Y", &[("S", "1")], &ctx).unwrap_err(),
        ObservationQueryError::ZeroProbabilityEvidence
    );
    assert_eq!(
        condition(&law, "Y", &[("Y", "1")], &ctx).unwrap_err(),
        ObservationQueryError::OutcomeObserved("Y".into())
    );
    assert!(matches!(
        condition(&law, "Y", &[("Q", "1")], &ctx).unwrap_err(),
        ObservationQueryError::UnknownVariable(_)
    ));
    assert!(matches!(
        condition(&law, "Y", &[("S", "7")], &ctx).unwrap_err(),
        ObservationQueryError::UnknownLevel { .. }
    ));
    assert_eq!(
        condition(&law, "Y", &[("S", "0"), ("S", "0")], &ctx).unwrap_err(),
        ObservationQueryError::RepeatedVariable("S".into())
    );
}

#[test]
fn the_conditional_is_invariant_to_the_order_of_the_evidence() {
    let ctx = ExecutionContext::for_tests(1);
    let law = confounded([0.2, 0.8]).law(&[], &ctx).unwrap();
    let a = condition(&law, "Y", &[("S", "1"), ("Z", "0")], &ctx).unwrap();
    let b = condition(&law, "Y", &[("Z", "0"), ("S", "1")], &ctx).unwrap();
    assert_eq!(a, b);
}

fn measured_model() -> ComposedFinite {
    ComposedFinite::compile(&spec_of(vec![
        table("C", &[], B, &[(&[], &[0.9, 0.1])]),
        table("Y", &[("C", B)], B, &[(&["0"], &[0.9, 0.1]), (&["1"], &[0.3, 0.7])]),
    ]))
    .unwrap()
}

#[test]
fn a_measured_signal_is_inverted_through_the_confusion_not_read_as_the_construct() {
    let ctx = ExecutionContext::for_tests(1);
    let law = measured_model().law(&[], &ctx).unwrap();
    // P(M = 1 | C): sensitivity 0.9 for C=1, 1 - specificity = 0.2 for C=0.
    let positive = condition_on_measured(&law, "Y", "C", &[], &[0.2, 0.9], &ctx).unwrap();
    near(positive.evidence_probability, 0.27);
    near(level(&positive.construct_posterior, "1"), 1.0 / 3.0);
    near(level(&positive.construct_posterior, "0"), 2.0 / 3.0);
    near(level(&positive.outcome, "1"), 0.3);

    // P(M = 0 | C): 0.8 for C=0, 0.1 for C=1.
    let negative = condition_on_measured(&law, "Y", "C", &[], &[0.8, 0.1], &ctx).unwrap();
    near(negative.evidence_probability, 0.73);
    near(level(&negative.construct_posterior, "1"), 1.0 / 73.0);
    near(level(&negative.outcome, "1"), 7.9 / 73.0);

    // The measured-as-construct read answers a different question.
    let as_construct = condition(&law, "Y", &[("C", "1")], &ctx).unwrap();
    near(level(&as_construct.levels, "1"), 0.7);
    assert!((level(&as_construct.levels, "1") - level(&positive.outcome, "1")).abs() > 0.3);

    // A perfect indicator reproduces conditioning on the construct.
    let perfect = condition_on_measured(&law, "Y", "C", &[], &[0.0, 1.0], &ctx).unwrap();
    near(level(&perfect.outcome, "1"), 0.7);
    near(level(&perfect.construct_posterior, "1"), 1.0);

    // The construct is its own outcome: the posterior itself.
    let posterior = condition_on_measured(&law, "C", "C", &[], &[0.2, 0.9], &ctx).unwrap();
    near(level(&posterior.outcome, "1"), 1.0 / 3.0);
}

#[test]
fn a_measured_signal_refuses_a_misaligned_likelihood_and_a_directly_observed_construct() {
    let ctx = ExecutionContext::for_tests(1);
    let law = measured_model().law(&[], &ctx).unwrap();
    assert!(matches!(
        condition_on_measured(&law, "Y", "C", &[], &[0.2], &ctx).unwrap_err(),
        ObservationQueryError::Invalid(_)
    ));
    assert!(matches!(
        condition_on_measured(&law, "Y", "C", &[], &[0.2, 1.5], &ctx).unwrap_err(),
        ObservationQueryError::Invalid(_)
    ));
    assert_eq!(
        condition_on_measured(&law, "Y", "C", &[("C", "1")], &[0.2, 0.9], &ctx).unwrap_err(),
        ObservationQueryError::ConstructObserved("C".into())
    );
    assert_eq!(
        condition_on_measured(&law, "Y", "C", &[], &[0.0, 0.0], &ctx).unwrap_err(),
        ObservationQueryError::ZeroProbabilityEvidence
    );
}

#[test]
fn threshold_crossing_lists_the_observations_that_cross_and_their_probability() {
    let ctx = ExecutionContext::for_tests(1);
    let law = confounded([0.2, 0.8]).law(&[], &ctx).unwrap();
    let up = threshold_crossing_observations(&law, "Y", "1", "S", &[], 0.5, Direction::Raise, &ctx)
        .unwrap();
    assert_eq!(up.levels.len(), 2);
    let s1 = up.levels.iter().find(|r| r.level == "1").unwrap();
    near(s1.conditional, 0.78);
    near(s1.probability, 0.5);
    assert!(s1.crosses);
    let s0 = up.levels.iter().find(|r| r.level == "0").unwrap();
    near(s0.conditional, 0.18);
    assert!(!s0.crosses);
    near(up.crossing_probability, 0.5);

    let down =
        threshold_crossing_observations(&law, "Y", "1", "S", &[], 0.2, Direction::Lower, &ctx)
            .unwrap();
    near(down.crossing_probability, 0.5);
    assert!(down.levels.iter().find(|r| r.level == "0").unwrap().crosses);

    // With the confounder held fixed the conditional is sharper: P(Y=1|S=1,Z=1)=0.9.
    let held = threshold_crossing_observations(
        &law,
        "Y",
        "1",
        "S",
        &[("Z", "1")],
        0.85,
        Direction::Raise,
        &ctx,
    )
    .unwrap();
    near(held.levels.iter().find(|r| r.level == "1").unwrap().conditional, 0.9);
    // P(S=1 | Z=1) = 0.8
    near(held.crossing_probability, 0.8);
}

#[test]
fn a_threshold_no_observation_reaches_is_refused_with_the_supremum_reachable() {
    let ctx = ExecutionContext::for_tests(1);
    let law = confounded([0.2, 0.8]).law(&[], &ctx).unwrap();
    let error =
        threshold_crossing_observations(&law, "Y", "1", "S", &[], 0.9, Direction::Raise, &ctx)
            .unwrap_err();
    match error {
        ObservationQueryError::LevelUnreachable { level, extremum, direction, bound } => {
            near(level, 0.9);
            near(extremum, 0.78);
            assert_eq!(direction, Direction::Raise);
            assert_eq!(bound, UnreachableBound::NoObservationCrosses);
        }
        other => panic!("expected an unreachable level, got {other:?}"),
    }
    let low =
        threshold_crossing_observations(&law, "Y", "1", "S", &[], 0.1, Direction::Lower, &ctx)
            .unwrap_err();
    match low {
        ObservationQueryError::LevelUnreachable { extremum, .. } => near(extremum, 0.18),
        other => panic!("expected an unreachable level, got {other:?}"),
    }
}

#[test]
fn finite_bayes_update_matches_the_hand_computed_posterior() {
    let model = FiniteBayesModel::new(
        vec!["ill".into(), "well".into()],
        vec![0.01, 0.99],
        vec!["neg".into(), "pos".into()],
        vec![0.1, 0.9, 0.95, 0.05],
    )
    .unwrap();
    // One positive: 0.009 / (0.009 + 0.0495) = 2/13, evidence 0.0585.
    let one = model.posterior(&["pos"]).unwrap();
    near(one.hypotheses[0].1, 2.0 / 13.0);
    near(one.hypotheses[1].1, 11.0 / 13.0);
    near(one.evidence_probability, 0.0585);
    // Predictive: (2/13)*0.9 + (11/13)*0.05 = 2.35/13.
    near(one.predictive.iter().find(|(l, _)| l == "pos").unwrap().1, 2.35 / 13.0);
    near(one.predictive.iter().map(|(_, p)| p).sum::<f64>(), 1.0);

    // Two positives: 0.0081 / (0.0081 + 0.002475).
    let two = model.posterior(&["pos", "pos"]).unwrap();
    near(two.hypotheses[0].1, 0.0081 / 0.010_575);
    near(two.evidence_probability, 0.010_575);

    // Exchangeable observations: order is irrelevant.
    let a = model.posterior(&["pos", "neg", "pos"]).unwrap();
    let b = model.posterior(&["neg", "pos", "pos"]).unwrap();
    assert_eq!(a, b);
    // Posterior ill = 0.01*0.9*0.9*0.1 / (that + 0.99*0.05*0.05*0.95)
    let ill = 0.01 * 0.9 * 0.9 * 0.1;
    let well = 0.99 * 0.05 * 0.05 * 0.95;
    near(a.hypotheses[0].1, ill / (ill + well));

    // No observations returns the prior.
    let none = model.posterior(&[]).unwrap();
    near(none.hypotheses[0].1, 0.01);
    near(none.evidence_probability, 1.0);

    // The count form is the same update, without materialising the stream.
    let counts = model.posterior_from_counts(&[1, 2]).unwrap();
    assert_eq!(counts, a);
    assert_eq!(model.posterior_from_counts(&[1]).unwrap_err().to_string(), "invalid declaration: one count per observation level is required");
    let long = model.posterior_from_counts(&[0, 5_000_000]).unwrap();
    assert!(long.hypotheses[0].1 > 1.0 - 1e-6);
}

#[test]
fn finite_bayes_model_refuses_malformed_declarations_and_impossible_evidence() {
    let names = || vec!["a".to_owned(), "b".to_owned()];
    assert_eq!(
        FiniteBayesModel::new(names(), vec![0.5, 0.4], names(), vec![0.5; 4]).unwrap_err().to_string(),
        "invalid declaration: the prior is not a distribution"
    );
    assert_eq!(
        FiniteBayesModel::new(names(), vec![0.5, 0.5], names(), vec![0.5, 0.4, 0.5, 0.5])
            .unwrap_err()
            .to_string(),
        "invalid declaration: a likelihood row is not a distribution over the observation levels"
    );
    assert_eq!(
        FiniteBayesModel::new(vec!["a".into(), "a".into()], vec![0.5, 0.5], names(), vec![0.5; 4])
            .unwrap_err()
            .to_string(),
        "invalid declaration: hypotheses must be distinct non-blank names"
    );
    let degenerate =
        FiniteBayesModel::new(names(), vec![0.5, 0.5], names(), vec![1.0, 0.0, 1.0, 0.0]).unwrap();
    assert_eq!(
        degenerate.posterior(&["b"]).unwrap_err(),
        ObservationQueryError::ZeroProbabilityEvidence
    );
    assert!(matches!(
        degenerate.posterior(&["zzz"]).unwrap_err(),
        ObservationQueryError::UnknownLevel { .. }
    ));
}

#[test]
fn conjugate_update_matches_the_closed_form() {
    let prior = BetaBernoulli::new(1.0, 1.0).unwrap();
    let posterior = prior.update(7, 3).unwrap();
    near(posterior.alpha(), 8.0);
    near(posterior.beta(), 4.0);
    near(posterior.mean(), 2.0 / 3.0);
    // Sequential and batch updates agree.
    let sequential = prior.update(3, 1).unwrap().update(4, 2).unwrap();
    near(sequential.alpha(), posterior.alpha());
    near(sequential.beta(), posterior.beta());
    assert_eq!(BetaBernoulli::new(0.0, 1.0).unwrap_err().to_string(), "invalid declaration: Beta shapes must be finite and positive");
    assert_eq!(BetaBernoulli::new(f64::NAN, 1.0).unwrap_err().to_string(), "invalid declaration: Beta shapes must be finite and positive");
    assert_eq!(
        BetaBernoulli::new(1.0, 1.0).unwrap().update(1_u64 << 53, 1).unwrap_err().to_string(),
        "invalid declaration: observation counts exceed 2^53"
    );
}

#[test]
fn beta_queries_remain_finite_for_large_valid_shapes() {
    let prior = BetaBernoulli::new(1.0e308, 1.0e308).unwrap();
    near(prior.mean(), 0.5);

    // The closed form used to overflow at alpha + beta, falsely making the
    // predictive probability zero and reporting the wrong budget extremum.
    match observations_required_conjugate(&prior, 0.6, Direction::Raise, 100).unwrap_err() {
        ObservationQueryError::LevelUnreachable { extremum, .. } => near(extremum, 0.5),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn the_required_bayes_factor_is_the_odds_ratio() {
    // odds 0.2 -> 0.25, odds 0.8 -> 4, factor 16.
    let r = required_bayes_factor(0.2, 0.8, Direction::Raise).unwrap();
    near(r.factor, 16.0);
    assert!(!r.already_satisfied);
    let l = required_bayes_factor(0.8, 0.2, Direction::Lower).unwrap();
    near(l.factor, 1.0 / 16.0);
    let satisfied = required_bayes_factor(0.8, 0.5, Direction::Raise).unwrap();
    assert!(satisfied.already_satisfied);
    near(satisfied.factor, 1.0);
    assert_eq!(required_bayes_factor(0.0, 0.5, Direction::Raise).unwrap_err().to_string(), "invalid declaration: the prior must lie in (0, 1)");
    assert_eq!(required_bayes_factor(0.5, 1.0, Direction::Raise).unwrap_err().to_string(), "invalid declaration: the level must lie in (0, 1)");
}

#[test]
fn observations_required_for_a_binary_signal_are_the_exact_minimum() {
    // Bayes factor 0.9/0.1 = 9; prior odds 1; level 0.9 is odds 9: one reading.
    let one =
        observations_required_binary_signal(0.5, 0.9, Direction::Raise, 0.9, 0.1, 100).unwrap();
    assert_eq!(one.observations, 1);
    near(one.value_at_requirement, 0.9);
    near(one.bayes_factor_per_observation.unwrap(), 9.0);

    // Prior 0.01, Bayes factor 0.9/0.05 = 18, level 0.5: odds 1/99 -> need 99.
    // 18^1 = 18 < 99 <= 324 = 18^2: two readings, posterior 324/423.
    let two =
        observations_required_binary_signal(0.01, 0.5, Direction::Raise, 0.9, 0.05, 100).unwrap();
    assert_eq!(two.observations, 2);
    near(two.value_at_requirement, (324.0 / 99.0) / (1.0 + 324.0 / 99.0));
    // The count is a minimum: one fewer does not reach the level.
    let one_fewer = (18.0 / 99.0) / (1.0 + 18.0 / 99.0);
    assert!(one_fewer < 0.5);

    // Lowering with negative readings: factor 0.1/0.9 = 1/9 takes 0.5 to 0.1.
    let lower =
        observations_required_binary_signal(0.5, 0.1, Direction::Lower, 0.9, 0.1, 100).unwrap();
    assert_eq!(lower.observations, 1);
    near(lower.value_at_requirement, 0.1);

    // Already past the level: zero observations.
    let done =
        observations_required_binary_signal(0.7, 0.5, Direction::Raise, 0.9, 0.1, 100).unwrap();
    assert!(done.already_satisfied);
    assert_eq!(done.observations, 0);

    // A signal that rules the complement out settles it in one reading.
    let sure =
        observations_required_binary_signal(0.1, 0.99, Direction::Raise, 0.5, 0.0, 10).unwrap();
    assert_eq!(sure.observations, 1);
    near(sure.value_at_requirement, 1.0);
}

#[test]
fn a_level_the_signal_cannot_reach_states_the_supremum_reachable() {
    // Uninformative signal: the prior is the supremum.
    let flat = observations_required_binary_signal(0.3, 0.9, Direction::Raise, 0.5, 0.5, 1000)
        .unwrap_err();
    match flat {
        ObservationQueryError::LevelUnreachable { extremum, bound, .. } => {
            near(extremum, 0.3);
            assert_eq!(bound, UnreachableBound::SignalDoesNotMove);
        }
        other => panic!("unexpected {other:?}"),
    }
    // A positive reading that is evidence *against* the hypothesis cannot raise it.
    let wrong = observations_required_binary_signal(0.3, 0.9, Direction::Raise, 0.2, 0.6, 1000)
        .unwrap_err();
    assert!(matches!(
        wrong,
        ObservationQueryError::LevelUnreachable { bound: UnreachableBound::SignalDoesNotMove, .. }
    ));
    // Bayes factor 1.5 and a budget of three readings: odds 1 * 1.5^3 = 3.375, p = 3.375/4.375.
    let budget =
        observations_required_binary_signal(0.5, 0.99, Direction::Raise, 0.6, 0.4, 3).unwrap_err();
    match budget {
        ObservationQueryError::LevelUnreachable { extremum, bound, .. } => {
            near(extremum, 3.375 / 4.375);
            assert_eq!(bound, UnreachableBound::ObservationBudget { max_observations: 3 });
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn observations_required_for_a_conjugate_model_match_the_closed_form() {
    let prior = BetaBernoulli::new(1.0, 1.0).unwrap();
    // (1 + n) / (2 + n) >= 0.9  <=>  n >= 8, and exactly 0.9 at n = 8.
    let up = observations_required_conjugate(&prior, 0.9, Direction::Raise, 1000).unwrap();
    assert_eq!(up.observations, 8);
    near(up.value_at_requirement, 0.9);
    // 1 / (2 + n) <= 0.1  <=>  n >= 8.
    let down = observations_required_conjugate(&prior, 0.1, Direction::Lower, 1000).unwrap();
    assert_eq!(down.observations, 8);
    near(down.value_at_requirement, 0.1);
    // Beta(3, 1): mean 0.75. (3 + n) / (4 + n) >= 0.95  <=>  n >= 16.
    let strong = BetaBernoulli::new(3.0, 1.0).unwrap();
    assert_eq!(
        observations_required_conjugate(&strong, 0.95, Direction::Raise, 1000)
            .unwrap()
            .observations,
        16
    );
    // Already satisfied.
    assert!(
        observations_required_conjugate(&strong, 0.5, Direction::Raise, 1000)
            .unwrap()
            .already_satisfied
    );
    // Budget: 98 successes are needed for 0.99 but only 50 are allowed.
    let error = observations_required_conjugate(&prior, 0.99, Direction::Raise, 50).unwrap_err();
    match error {
        ObservationQueryError::LevelUnreachable { extremum, bound, .. } => {
            near(extremum, 51.0 / 52.0);
            assert_eq!(bound, UnreachableBound::ObservationBudget { max_observations: 50 });
        }
        other => panic!("unexpected {other:?}"),
    }
    // The exact count is reachable once the budget allows it.
    assert_eq!(
        observations_required_conjugate(&prior, 0.99, Direction::Raise, 98).unwrap().observations,
        98
    );
}

fn long_chain(width: usize) -> ComposedFinite {
    let name = |i: usize| format!("v{i:02}");
    let mut mechanisms = vec![table(&name(0), &[], B, &[(&[], &[0.5, 0.5])])];
    for i in 1..width {
        let parent = name(i - 1);
        mechanisms.push(table(
            &name(i),
            &[(parent.as_str(), B)],
            B,
            &[(&["0"], &[0.7, 0.3]), (&["1"], &[0.4, 0.6])],
        ));
    }
    ComposedFinite::compile(&spec_of(mechanisms)).unwrap()
}

#[test]
fn queries_observe_cancellation_on_a_large_law() {
    let ctx = ExecutionContext::for_tests(1);
    // 2^13 joint states: above the cancellation polling interval.
    let model = long_chain(13);
    let law = model.law(&[], &ctx).unwrap();
    near(condition(&law, "v12", &[("v00", "1")], &ctx).unwrap().evidence_probability, 0.5);
    ctx.cancellation.cancel();
    assert_eq!(
        condition(&law, "v12", &[("v00", "1")], &ctx).unwrap_err(),
        ObservationQueryError::Cancelled
    );
    assert_eq!(
        conditioning_and_intervention(&model, "v12", &[("v00", "1")], &ctx).unwrap_err(),
        ObservationQueryError::Cancelled
    );
}

// ------------------------------------------------ continuous (Gaussian) evidence

use antecedent_model::compose::{ComposedLinearGaussian, GaussianConditioningError};
use antecedent_model::observation_query::{condition_after_intervention, condition_gaussian};
use antecedent_model::observation_uncertainty::{
    GaussianSufficient, RedrawPrior, RedrawTarget, RetainedMechanism, TableCountRow, TableCounts,
    redraw_conditional,
};

fn real_binding(var: &str) -> ParentBinding {
    ParentBinding {
        variable: var.into(),
        units: "u".into(),
        transform: "identity".into(),
        domain: Domain::Real,
    }
}

fn lg(var: &str, intercept: f64, sigma: f64, parents: &[(&str, f64)]) -> MechanismDescriptor {
    MechanismDescriptor {
        variable: var.into(),
        form: MechanismForm::LinearGaussian {
            intercept,
            sigma,
            coefficients: parents.iter().map(|(p, c)| ((*p).to_owned(), *c)).collect(),
        },
        parents: parents.iter().map(|(p, _)| real_binding(p)).collect(),
        conditioning_set: parents.iter().map(|(p, _)| (*p).to_owned()).collect(),
        output_units: "u".into(),
        output_transform: "identity".into(),
        output_support: Domain::Real,
        context: MechanismContext { population: "pop".into(), regime: "observational".into() },
        provenance: MechanismProvenance {
            origin: MechanismOrigin::Fitted,
            artifact_id: format!("fit:{var}"),
            evidence_id: format!("data:{var}"),
        },
        latents: Vec::new(),
        uncertainty: None,
    }
}

/// X ~ N(1, 4); M = 0.5 + 2 X + e (sd 1); Y = -1 + 0.5 X + 3 M + 0.5 e.
/// Joint: mean (1, 2.5, 7); var X 4, M 17, Y 178.25; cov XM 8, XY 26, MY 55.
fn gaussian_chain() -> ComposedLinearGaussian {
    ComposedLinearGaussian::compile(&spec_of(vec![
        lg("X", 1.0, 2.0, &[]),
        lg("M", 0.5, 1.0, &[("X", 2.0)]),
        lg("Y", -1.0, 0.5, &[("X", 0.5), ("M", 3.0)]),
    ]))
    .unwrap()
}

#[test]
fn schur_complement_conditioning_matches_closed_forms() {
    let ctx = ExecutionContext::for_tests(1);
    let model = gaussian_chain();

    // One observed coordinate: Y | M = 3 has mean 7 + (55 / 17)(3 - 2.5) and variance
    // 178.25 - 55^2 / 17.
    let one = condition_gaussian(&model, "Y", &[], &[("M", 3.0)], &ctx).unwrap();
    near(one.mean, 7.0 + 55.0 / 17.0 * 0.5);
    near(one.variance, 178.25 - 3025.0 / 17.0);
    // The density of the evidence is the marginal N(2.5, 17) at 3.
    let expected_density = -0.5 * ((2.0 * std::f64::consts::PI).ln() + 17.0_f64.ln() + 0.25 / 17.0);
    near(one.evidence_log_density.unwrap(), expected_density);

    // Conditioning on both parents of Y returns the structural equation: N(9, 0.25).
    let parents = condition_gaussian(&model, "Y", &[], &[("X", 2.0), ("M", 3.0)], &ctx).unwrap();
    near(parents.mean, -1.0 + 1.0 + 9.0);
    near(parents.variance, 0.25);

    // Two observed coordinates, M given (X = 2, Y = 10), by the explicit 2x2 inverse:
    // det = 37, S_oo^-1 r = (100.25, -14) / 37, mean = 2.5 + 32 / 37, variance = 1 / 37.
    let two = condition_gaussian(&model, "M", &[], &[("X", 2.0), ("Y", 10.0)], &ctx).unwrap();
    near(two.mean, 2.5 + 32.0 / 37.0);
    near(two.variance, 1.0 / 37.0);

    // The order in which observed values are listed is not part of the answer.
    let reversed = condition_gaussian(&model, "M", &[], &[("Y", 10.0), ("X", 2.0)], &ctx).unwrap();
    assert_eq!(two, reversed);

    // Nothing observed is the marginal law, with no evidence density.
    let none = condition_gaussian(&model, "Y", &[], &[], &ctx).unwrap();
    near(none.mean, 7.0);
    near(none.variance, 178.25);
    assert!(none.evidence_log_density.is_none());
}

#[test]
fn observing_after_an_intervention_conditions_the_cut_graph() {
    let ctx = ExecutionContext::for_tests(1);
    let model = gaussian_chain();
    // do(X = 2): M ~ N(4.5, 1), Y ~ N(13.5, 9.25), cov(M, Y) = 3. Observing M = 5 gives
    // Y | ... of mean 13.5 + 3 (0.5) = 15 and variance 9.25 - 9 = 0.25, which is the
    // structural equation at X = 2, M = 5: -1 + 1 + 15 and 0.5^2.
    let mixed = condition_gaussian(&model, "Y", &[("X", 2.0)], &[("M", 5.0)], &ctx).unwrap();
    near(mixed.mean, 15.0);
    near(mixed.variance, 0.25);
    // The observational conditional on the same M is 7 + (55/17)(2.5) = 15.088...; at X = 2 the
    // two nearly agree (X = 2 is close to what M = 5 implies), so cut the graph elsewhere:
    // do(X = -3), M = 5 is the structural equation -1 - 1.5 + 15 = 12.5 with variance 0.25.
    let seen = condition_gaussian(&model, "Y", &[], &[("M", 5.0)], &ctx).unwrap();
    near(seen.mean, 7.0 + 55.0 / 17.0 * 2.5);
    let cut = condition_gaussian(&model, "Y", &[("X", -3.0)], &[("M", 5.0)], &ctx).unwrap();
    near(cut.mean, 12.5);
    near(cut.variance, 0.25);
    assert!((seen.mean - cut.mean).abs() > 2.0);
    // Intervention alone is the interventional law.
    let alone = condition_gaussian(&model, "Y", &[("X", 2.0)], &[], &ctx).unwrap();
    near(alone.mean, 13.5);
    near(alone.variance, 9.25);
}

#[test]
fn gaussian_conditioning_refusals_are_typed() {
    let ctx = ExecutionContext::for_tests(1);
    let model = gaussian_chain();
    assert_eq!(
        condition_gaussian(&model, "Y", &[("X", 2.0)], &[("X", 2.0)], &ctx).unwrap_err(),
        ObservationQueryError::ObservedAndIntervened("X".into())
    );
    assert_eq!(
        condition_gaussian(&model, "Y", &[], &[("Y", 1.0)], &ctx).unwrap_err(),
        ObservationQueryError::OutcomeObserved("Y".into())
    );
    assert_eq!(
        condition_gaussian(&model, "Y", &[], &[("Q", 1.0)], &ctx).unwrap_err(),
        ObservationQueryError::UnknownVariable("Q".into())
    );
    assert_eq!(
        condition_gaussian(&model, "Y", &[], &[("M", 1.0), ("M", 2.0)], &ctx).unwrap_err(),
        ObservationQueryError::RepeatedVariable("M".into())
    );
    assert_eq!(
        condition_gaussian(&model, "Y", &[], &[("M", f64::NAN)], &ctx).unwrap_err(),
        ObservationQueryError::Invalid("an observed value must be finite")
    );
    // An intervened coordinate has zero variance: conditioning on it is singular.
    let law = model.law(&[("X", 2.0)]).unwrap();
    assert_eq!(
        law.condition(&[("X", 2.0)]).unwrap_err(),
        GaussianConditioningError::SingularEvidence
    );
    ctx.cancellation.cancel();
    assert_eq!(
        condition_gaussian(&model, "Y", &[], &[("M", 1.0)], &ctx).unwrap_err(),
        ObservationQueryError::Cancelled
    );
}

// ------------------------------------------ observation after intervention (finite)

#[test]
fn observation_after_an_intervention_matches_hand_enumeration() {
    let ctx = ExecutionContext::for_tests(1);
    let model = confounded([0.2, 0.8]);
    // P(Y=1 | do(S=1), Z=1) = 0.9: the cut graph leaves Z at its prior and reads the cell.
    let seen_z =
        condition_after_intervention(&model, "Y", &[("S", "1")], &[("Z", "1")], &ctx).unwrap();
    near(level(&seen_z.levels, "1"), 0.9);
    near(seen_z.evidence_probability, 0.5);
    // P(Z=1 | do(S=1), Y=1): a descendant of the intervention is informative about the cause
    // that was left alone. By enumeration over z with S fixed at 1:
    //   weight(z) = P(z) P(Y=1 | S=1, z) -> 0.5*0.3 = 0.15 and 0.5*0.9 = 0.45, so 0.75.
    let weights: Vec<f64> = [0.3_f64, 0.9].iter().map(|p| 0.5 * p).collect();
    let after =
        condition_after_intervention(&model, "Z", &[("S", "1")], &[("Y", "1")], &ctx).unwrap();
    near(level(&after.levels, "1"), weights[1] / weights.iter().sum::<f64>());
    near(after.evidence_probability, 0.6);
    // The observational conditional on the same recorded values keeps the S-Z association:
    // P(Z=1 | S=1, Y=1) = (0.5*0.8*0.9) / (0.5*0.8*0.9 + 0.5*0.2*0.3) = 0.36 / 0.39.
    let law = model.law(&[], &ctx).unwrap();
    let observed = condition(&law, "Z", &[("S", "1"), ("Y", "1")], &ctx).unwrap();
    near(level(&observed.levels, "1"), 0.36 / 0.39);
    assert!((level(&observed.levels, "1") - level(&after.levels, "1")).abs() > 0.1);
    // No recorded values is the interventional law.
    let none = condition_after_intervention(&model, "Y", &[("S", "1")], &[], &ctx).unwrap();
    near(level(&none.levels, "1"), 0.6);
    // A variable that is both recorded and set is refused.
    assert_eq!(
        condition_after_intervention(&model, "Y", &[("S", "1")], &[("S", "1")], &ctx).unwrap_err(),
        ObservationQueryError::ObservedAndIntervened("S".into())
    );
}

// ------------------------------------------------ parameter uncertainty by redraw

/// `y_i = 1 + 2 x_i + 0.5 sin(7 i)` at `x_i = i / 10`, `i < 50`.
fn regression_data() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..50_u32).map(|i| f64::from(i) / 10.0).collect();
    let y: Vec<f64> =
        x.iter().enumerate().map(|(i, xi)| 1.0 + 2.0 * xi + 0.5 * (7.0 * i as f64).sin()).collect();
    (x, y)
}

fn regression_spec() -> ComposedModelSpec {
    spec_of(vec![lg("X", 0.0, 1.0, &[]), lg("Y", 1.0, 0.5, &[("X", 2.0)])])
}

#[test]
fn normal_inverse_gamma_redraw_matches_the_student_t_posterior() {
    let ctx = ExecutionContext::for_tests(3);
    let (x, y) = regression_data();
    let stats = GaussianSufficient::from_columns("Y", &[("X", &x)], &y).unwrap();
    let prior = RedrawPrior::default();
    let target = RedrawTarget::Gaussian {
        outcome: "Y".into(),
        interventions: vec![("X".into(), 3.0)],
        observed: vec![],
    };
    let draws = redraw_conditional(
        &regression_spec(),
        &target,
        &[RetainedMechanism::Gaussian(stats)],
        &prior,
        20_000,
        7,
        &ctx,
    )
    .unwrap();
    assert_eq!(draws.quantities, ["conditional_mean", "conditional_variance"]);
    assert_eq!(draws.draws, 20_000);
    // The point value is the composition's own numbers: 1 + 2 * 3.
    near(draws.point[0], 7.0);

    // Independent closed form: normal equations of the 2x2 regression with the ridge, then
    // the Student-t law of w'beta for w = (1, 3) with 2 a_n degrees of freedom.
    let n = 50.0;
    let sx: f64 = x.iter().sum();
    let sxx: f64 = x.iter().map(|v| v * v).sum();
    let sy: f64 = y.iter().sum();
    let sxy: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
    let syy: f64 = y.iter().map(|v| v * v).sum();
    let (a11, a12, a22) = (n + prior.coefficient_precision, sx, sxx + prior.coefficient_precision);
    let det = a11 * a22 - a12 * a12;
    let m0 = (a22 * sy - a12 * sxy) / det;
    let m1 = (-a12 * sy + a11 * sxy) / det;
    let shape = prior.variance_shape + n / 2.0;
    let scale = prior.variance_scale + 0.5 * (syy - m0 * sy - m1 * sxy);
    let w = [1.0, 3.0];
    let quad = (a22 * w[0] * w[0] - 2.0 * a12 * w[0] * w[1] + a11 * w[1] * w[1]) / det;
    let location = m0 + 3.0 * m1;
    let df = 2.0 * shape;
    let t_sd = (scale / shape * quad).sqrt() * (df / (df - 2.0)).sqrt();

    let mean = draws.mean(0);
    let sd = draws.standard_deviation(0);
    assert!((mean - location).abs() < 5.0 * t_sd / (20_000.0_f64).sqrt(), "{mean} vs {location}");
    assert!((sd / t_sd - 1.0).abs() < 0.05, "{sd} vs {t_sd}");
    // The conditional variance of Y under do(X) is sigma^2 ~ InvGamma(a_n, b_n): mean b / (a - 1).
    let variance_mean = draws.mean(1);
    let expected = scale / (shape - 1.0);
    assert!((variance_mean / expected - 1.0).abs() < 0.03, "{variance_mean} vs {expected}");
}

#[test]
fn redraw_is_deterministic_and_the_rows_are_one_joint_realization() {
    let ctx = ExecutionContext::for_tests(3);
    let (x, y) = regression_data();
    let stats = GaussianSufficient::from_columns("Y", &[("X", &x)], &y).unwrap();
    let target = RedrawTarget::Gaussian {
        outcome: "Y".into(),
        interventions: vec![],
        observed: vec![("X".into(), 1.0)],
    };
    let run = |seed: u64, draws: u32| {
        redraw_conditional(
            &regression_spec(),
            &target,
            &[RetainedMechanism::Gaussian(stats.clone())],
            &RedrawPrior::default(),
            draws,
            seed,
            &ctx,
        )
        .unwrap()
    };
    let a = run(11, 400);
    let b = run(11, 400);
    assert_eq!(a, b);
    assert_ne!(a.values, run(12, 400).values);
    // Draw i of a longer run begins with the same rows: a draw depends on the seed and its
    // index only.
    let longer = run(11, 800);
    assert_eq!(a.values[..], longer.values[..a.values.len()]);
    assert_eq!(a.values.len(), 400 * 2);
}

#[test]
fn dirichlet_redraw_of_one_table_matches_the_beta_posterior() {
    let ctx = ExecutionContext::for_tests(3);
    let spec = spec_of(vec![table("Y", &[], B, &[(&[], &[0.3, 0.7])])]);
    let counts = TableCounts {
        variable: "Y".into(),
        rows: vec![TableCountRow {
            parent_levels: vec![],
            counts: vec![("0".into(), 30), ("1".into(), 70)],
        }],
    };
    let target =
        RedrawTarget::Finite { outcome: "Y".into(), interventions: vec![], observed: vec![] };
    let draws = redraw_conditional(
        &spec,
        &target,
        &[RetainedMechanism::Table(counts)],
        &RedrawPrior::default(),
        20_000,
        5,
        &ctx,
    )
    .unwrap();
    assert_eq!(draws.quantities, ["probability:0", "probability:1"]);
    near(draws.point[1], 0.7);
    // P(Y=1) ~ Beta(71, 31): mean 71/102, variance 71 * 31 / (102^2 * 103).
    let mean = 71.0 / 102.0;
    let sd = (71.0_f64 * 31.0 / (102.0 * 102.0 * 103.0)).sqrt();
    assert!((draws.mean(1) - mean).abs() < 5.0 * sd / (20_000.0_f64).sqrt());
    assert!((draws.standard_deviation(1) / sd - 1.0).abs() < 0.04);
    // The two levels of one draw are one distribution: every row sums to one.
    for row in draws.values.chunks(2) {
        assert!((row[0] + row[1] - 1.0).abs() < 1e-12);
    }
}

#[test]
fn redraw_refuses_mismatched_retained_data_and_bad_counts() {
    let ctx = ExecutionContext::for_tests(3);
    let (x, y) = regression_data();
    let target =
        RedrawTarget::Gaussian { outcome: "Y".into(), interventions: vec![], observed: vec![] };
    let draw = |retained: RetainedMechanism, draws: u32, prior: RedrawPrior| {
        redraw_conditional(&regression_spec(), &target, &[retained], &prior, draws, 1, &ctx)
    };
    let good = GaussianSufficient::from_columns("Y", &[("X", &x)], &y).unwrap();
    // Regressors that are not the declared parents.
    let wrong = GaussianSufficient::from_columns("Y", &[], &y).unwrap();
    assert!(matches!(
        draw(RetainedMechanism::Gaussian(wrong), 10, RedrawPrior::default()),
        Err(ObservationQueryError::Invalid(_))
    ));
    // Data of another family than the mechanism.
    let counts = TableCounts { variable: "Y".into(), rows: vec![] };
    assert!(matches!(
        draw(RetainedMechanism::Table(counts), 10, RedrawPrior::default()),
        Err(ObservationQueryError::Invalid(_))
    ));
    // A draw count outside the supported range and a prior outside its range.
    assert_eq!(
        draw(RetainedMechanism::Gaussian(good.clone()), 1, RedrawPrior::default())
            .unwrap_err()
            .to_string(),
        "invalid declaration: the draw count must lie between the supported minimum and maximum"
    );
    let bad_prior = RedrawPrior { variance_scale: 0.0, ..RedrawPrior::default() };
    assert_eq!(
        draw(RetainedMechanism::Gaussian(good), 10, bad_prior).unwrap_err().to_string(),
        "invalid declaration: every redraw prior parameter must be finite and positive"
    );
    // Retained data of a variable the composition does not have.
    let stranger = GaussianSufficient::from_columns("Q", &[], &y).unwrap();
    assert_eq!(
        draw(RetainedMechanism::Gaussian(stranger), 10, RedrawPrior::default()).unwrap_err(),
        ObservationQueryError::UnknownVariable("Q".into())
    );
}

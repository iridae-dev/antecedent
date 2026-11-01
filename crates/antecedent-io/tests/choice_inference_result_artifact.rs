//! `choice_inference_result_v1`: choice-study results recomputed from the retained respondents
//! and sources by a fresh consumer, bound to the choice contract, with every semantic edit
//! refused.
//!
//! Oracles: the Dirichlet posterior of pooled counts, whose shapes and means are closed forms
//! (prior plus discounted source plus counts), and a symmetric history whose
//! probability-best is exactly one third per option. The hierarchical routes are checked against
//! the stats crate's frozen mpmath/cubature oracle (`hierarchical_choice_reference`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::adaptive_design::{
    AssignmentFamily, ChoiceDesignContract, ChoiceDesignDeclaration, ChoiceLook, ChoiceRespondent,
    InferenceParadigm, MonitoringStrategy, OutcomeStructure, ProcedureCell, RetainedChoiceSource,
    StoppingRuleDeclaration, StoppingRuleKind, StoppingSpec,
};
use antecedent_core::ExecutionContext;
use antecedent_io::choice_inference_result_artifact::{
    ChoiceInferenceResultArtifact, ChoiceInferenceResultArtifactError, ChoiceResultSpec,
    compute_choice_result,
};

type Artifact = ChoiceInferenceResultArtifact;
type Error = ChoiceInferenceResultArtifactError;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: expected {expected}, got {actual} (tolerance {tolerance})"
    );
}

fn labels() -> Vec<String> {
    ["a", "b", "c"].map(String::from).to_vec()
}

fn cell(outcome: OutcomeStructure, looks: bool) -> ProcedureCell {
    ProcedureCell {
        assignment: AssignmentFamily::FixedRandomization,
        outcome,
        monitoring: if looks {
            MonitoringStrategy::FixedInterimLooks
        } else {
            MonitoringStrategy::FinalAnalysisOnly
        },
        stopping: if looks {
            StoppingRuleKind::DecisionTheoretic
        } else {
            StoppingRuleKind::NoStopping
        },
    }
}

fn rule(looks: bool) -> StoppingRuleDeclaration {
    if looks {
        StoppingRuleDeclaration {
            kind: StoppingRuleKind::DecisionTheoretic,
            spec: StoppingSpec::PosteriorExpectedLoss { loss_threshold: 0.2 },
            ended_by_rule: false,
        }
    } else {
        StoppingRuleDeclaration {
            kind: StoppingRuleKind::NoStopping,
            spec: StoppingSpec::NoStopping,
            ended_by_rule: false,
        }
    }
}

fn source() -> RetainedChoiceSource {
    RetainedChoiceSource {
        label: "wave0".into(),
        option_labels: labels(),
        meaning: "chosen in the pricing task".into(),
        population: "customers".into(),
        counts: vec![10.0, 5.0, 5.0],
        observation_ids: (0..20).map(|i| format!("old{i}#0")).collect(),
        discount: 0.5,
        transfer_rationale: None,
    }
}

/// Six respondents with one choice each: four `a`, one `b`, one `c`.
fn categorical(looks: bool) -> ChoiceDesignDeclaration {
    let picks = [0, 0, 0, 0, 1, 2];
    ChoiceDesignDeclaration {
        label: "pricing".into(),
        cell: cell(OutcomeStructure::CategoricalChoice, looks),
        option_labels: labels(),
        meaning: "chosen in the pricing task".into(),
        population: "customers".into(),
        respondents: picks
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut counts = vec![0, 0, 0];
                counts[*p] = 1;
                ChoiceRespondent { id: format!("r{i}"), counts }
            })
            .collect(),
        initial_prior: vec![1.0; 3],
        retained_sources: vec![source()],
        looks: Some(if looks {
            vec![ChoiceLook { respondents: 3 }, ChoiceLook { respondents: 6 }]
        } else {
            vec![ChoiceLook { respondents: 6 }]
        }),
        stopping_rule: Some(rule(looks)),
    }
}

/// Six respondents choosing three times each.
fn repeated(looks: bool) -> ChoiceDesignDeclaration {
    let counts = [[3, 0, 0], [2, 1, 0], [2, 0, 1], [3, 0, 0], [1, 1, 1], [2, 1, 0]];
    let mut d = categorical(looks);
    d.cell = cell(OutcomeStructure::RepeatedChoice, looks);
    d.retained_sources.clear();
    d.respondents = counts
        .iter()
        .enumerate()
        .map(|(i, c)| ChoiceRespondent { id: format!("r{i}"), counts: c.to_vec() })
        .collect();
    d
}

fn hierarchical(look: u32) -> ChoiceResultSpec {
    ChoiceResultSpec::Hierarchical {
        look,
        shapes: vec![1.0; 3],
        log_concentration_mean: 5f64.ln(),
        log_concentration_sd: 1.0,
    }
}

fn contract(d: ChoiceDesignDeclaration) -> ChoiceDesignContract {
    ChoiceDesignContract::new(d).expect("complete")
}

#[test]
fn the_posterior_is_the_dirichlet_closed_form_on_the_qualified_prior() {
    let c = contract(categorical(false));
    let result = compute_choice_result(&c, &ChoiceResultSpec::Posterior { look: 0 }, &ctx())
        .expect("posterior");
    assert_eq!(result.paradigm, InferenceParadigm::BayesianPosterior);
    // prior 1 + 0.5 * (10, 5, 5) + counts (4, 1, 1)
    let alpha = [10.0, 4.5, 4.5];
    for k in 0..3 {
        close(result.values[4 * k], alpha[k], 1e-12, "alpha");
        close(result.values[4 * k + 1], alpha[k] / 19.0, 1e-12, "mean");
    }
    let pbest: f64 = (0..3).map(|k| result.values[4 * k + 2]).sum();
    close(pbest, 1.0, 1e-9, "probability best sums to one");
    close(result.values[12], 10.0, 1e-12, "borrowed observations");
    close(result.values[13], 0.0, 0.0, "no conflict");
    assert_eq!(result.closed_guarantee, "choice_inference.posterior_interval_closed");
}

#[test]
fn a_symmetric_history_has_equal_probability_best() {
    let mut d = categorical(false);
    d.retained_sources.clear();
    d.respondents.truncate(3);
    d.respondents[0].counts = vec![1, 0, 0];
    d.respondents[1].counts = vec![0, 1, 0];
    d.respondents[2].counts = vec![0, 0, 1];
    d.looks = Some(vec![ChoiceLook { respondents: 3 }]);
    let result =
        compute_choice_result(&contract(d), &ChoiceResultSpec::Posterior { look: 0 }, &ctx())
            .unwrap();
    for k in 0..3 {
        close(result.values[4 * k + 2], 1.0 / 3.0, 1e-9, "probability best");
    }
}

#[test]
fn every_route_seals_and_a_fresh_consumer_replays_it() {
    let cases: Vec<(ChoiceDesignContract, ChoiceResultSpec)> = vec![
        (contract(categorical(false)), ChoiceResultSpec::Posterior { look: 0 }),
        (contract(categorical(true)), ChoiceResultSpec::Posterior { look: 1 }),
        (contract(categorical(true)), ChoiceResultSpec::Stopping { look: 0 }),
        (contract(repeated(false)), hierarchical(0)),
        (
            contract(repeated(true)),
            ChoiceResultSpec::HierarchicalStopping {
                look: 1,
                shapes: vec![1.0; 3],
                log_concentration_mean: 5f64.ln(),
                log_concentration_sd: 1.0,
            },
        ),
    ];
    for (c, spec) in &cases {
        let bytes = Artifact::to_bytes(c, spec, "choice-1", &ctx()).expect("seal");
        let consumed = Artifact::from_bytes(&bytes, None, &ctx()).expect("replay");
        assert_eq!(consumed.spec(), spec);
        assert_eq!(consumed.contract().identity(), c.identity());
        let retained = Artifact::from_bytes(&bytes, Some(consumed.identity()), &ctx());
        assert!(retained.is_ok());
        assert!(matches!(
            Artifact::from_bytes(&bytes, Some("0".repeat(64).as_str()), &ctx()),
            Err(Error::IdentityMismatch { field: "retained_identity" })
        ));
    }
}

#[test]
fn a_stopping_decision_is_the_contracts_rule_on_the_posterior() {
    let c = contract(categorical(true));
    let at_three =
        compute_choice_result(&c, &ChoiceResultSpec::Stopping { look: 0 }, &ctx()).unwrap();
    let at_six =
        compute_choice_result(&c, &ChoiceResultSpec::Stopping { look: 1 }, &ctx()).unwrap();
    assert_eq!(at_three.values[0], 0.0);
    assert_eq!(at_six.values[0], 0.0);
    // the decision is the declared threshold applied to the posterior expected loss
    for r in [&at_three, &at_six] {
        assert_eq!(r.values[3] == 1.0, r.values[2] <= 0.2 + 1e-12);
    }
    assert_eq!(at_six.closed_guarantee, "choice_inference.stopping_reliability_closed");
}

#[test]
fn a_route_is_refused_for_the_wrong_outcome_or_an_unlicensed_paradigm() {
    let by_outcome =
        compute_choice_result(&contract(categorical(false)), &hierarchical(0), &ctx()).unwrap_err();
    assert_eq!(by_outcome.detail, "choice_inference.route_outcome_mismatch");
    let by_cell = compute_choice_result(
        &contract(categorical(false)),
        &ChoiceResultSpec::Stopping { look: 0 },
        &ctx(),
    )
    .unwrap_err();
    assert_ne!(by_cell.detail, "");
    assert!(
        compute_choice_result(
            &contract(categorical(true)),
            &ChoiceResultSpec::Posterior { look: 9 },
            &ctx()
        )
        .is_err()
    );
}

fn sealed(
    c: &ChoiceDesignContract,
    spec: &ChoiceResultSpec,
) -> antecedent_io::choice_inference_result_artifact::ChoiceResultMeta {
    Artifact::seal(c, spec, &ctx()).unwrap().1
}

#[test]
fn resealed_semantic_edits_are_refused() {
    let c = contract(categorical(true));
    let base = sealed(&c, &ChoiceResultSpec::Stopping { look: 1 });
    assert!(Artifact::from_meta(&base, None, &ctx()).is_ok());

    // a stored value
    let mut m = base.clone();
    m.value_bits[1] = 0.5f64.to_bits();
    Artifact::reseal_identity(&mut m);
    assert!(matches!(Artifact::from_meta(&m, None, &ctx()), Err(Error::ResultMismatch(_))));

    // a respondent's choice: the contract identity moves
    let mut m = base.clone();
    m.declaration.respondents[0].counts = vec![0, 1, 0];
    Artifact::reseal_identity(&mut m);
    assert!(matches!(
        Artifact::from_meta(&m, None, &ctx()),
        Err(Error::IdentityMismatch { field: "contract_identity" })
    ));

    // the stopping rule
    let mut m = base.clone();
    m.declaration.stopping_rule.as_mut().unwrap().parameter_bits = Some(0.3f64.to_bits());
    Artifact::reseal_identity(&mut m);
    assert!(Artifact::from_meta(&m, None, &ctx()).is_err());

    // a retained source's discount
    let mut m = base.clone();
    m.declaration.sources[0].discount_bits = 1.0f64.to_bits();
    Artifact::reseal_identity(&mut m);
    assert!(matches!(
        Artifact::from_meta(&m, None, &ctx()),
        Err(Error::IdentityMismatch { field: "contract_identity" })
    ));

    // the claim and the calibration status
    let mut m = base.clone();
    m.calibration = "measured".into();
    Artifact::reseal_identity(&mut m);
    assert!(matches!(Artifact::from_meta(&m, None, &ctx()), Err(Error::UnsupportedSemantics(_))));

    // an unresealed edit
    let mut m = base;
    m.value_bits[0] = 7.0f64.to_bits();
    assert!(matches!(
        Artifact::from_meta(&m, None, &ctx()),
        Err(Error::ResultMismatch(_) | Error::IdentityMismatch { .. })
    ));
}

#[test]
fn a_source_on_the_respondents_rows_is_refused_at_consumption() {
    let c = contract(categorical(false));
    let base = sealed(&c, &ChoiceResultSpec::Posterior { look: 0 });
    let mut m = base;
    m.declaration.sources[0].observation_ids[0] = "r0#0".into();
    Artifact::reseal_identity(&mut m);
    match Artifact::from_meta(&m, None, &ctx()) {
        Err(Error::Refused { detail, .. }) => {
            assert_eq!(detail, "choice_inference.source_overlaps_respondents");
        }
        other => panic!("expected the ledger's refusal, got {other:?}"),
    }
}

#[test]
fn a_conflicting_source_is_reported_in_the_values_not_hidden() {
    let mut d = categorical(false);
    d.retained_sources[0].counts = vec![1.0, 1.0, 60.0];
    d.retained_sources[0].discount = 1.0;
    let mut other = source();
    other.label = "wave1".into();
    other.counts = vec![60.0, 1.0, 1.0];
    other.discount = 1.0;
    other.observation_ids = (0..20).map(|i| format!("older{i}#0")).collect();
    d.retained_sources.push(other);
    let result =
        compute_choice_result(&contract(d), &ChoiceResultSpec::Posterior { look: 0 }, &ctx())
            .unwrap();
    assert!(result.values[13] >= 1.0, "a conflicting source is counted");
    assert!(result.values[14] > std::f64::consts::LN_10);
}

#[test]
fn the_hierarchical_posterior_carries_the_respondents_dependence() {
    // Pooled counts as a Dirichlet would overstate the information of repeated choosers.
    let c = contract(repeated(false));
    let h = compute_choice_result(&c, &hierarchical(0), &ctx()).unwrap();
    let mean: f64 = (0..3).map(|k| h.values[3 * k]).sum();
    close(mean, 1.0, 1e-8, "population shares sum to one");
    assert_eq!(h.closed_guarantee, "choice_inference.hierarchical_interval_closed");
    // pooled counts (13, 3, 2) of 18 choices: the pooled posterior is more certain `a` is best
    let mut pooled = categorical(false);
    pooled.retained_sources.clear();
    pooled.respondents = (0..18)
        .map(|i| {
            let mut counts = vec![0, 0, 0];
            counts[if i < 13 {
                0
            } else if i < 16 {
                1
            } else {
                2
            }] = 1;
            ChoiceRespondent { id: format!("p{i}"), counts }
        })
        .collect();
    pooled.looks = Some(vec![ChoiceLook { respondents: 18 }]);
    let p =
        compute_choice_result(&contract(pooled), &ChoiceResultSpec::Posterior { look: 0 }, &ctx())
            .unwrap();
    assert!(p.values[2] > h.values[1], "pooling treats repeated choices as independent");
}

fn uniform_final_only() -> ChoiceDesignDeclaration {
    let mut d = categorical(false);
    d.retained_sources.clear();
    d
}

#[test]
fn a_measured_interval_is_refused_while_no_coverage_record_attests_the_cell() {
    let bytes = Artifact::to_bytes(
        &contract(uniform_final_only()),
        &ChoiceResultSpec::Posterior { look: 0 },
        "choice-measured",
        &ctx(),
    )
    .unwrap();
    let refused = Artifact::measure_interval(&bytes, &ctx()).expect_err("no record");
    let text = format!("{refused:?}");
    assert!(
        text.contains("cell_not_licensed") || text.contains("not_measured"),
        "an unmeasured cell refuses rather than reports an interval: {text}"
    );
}

#[test]
fn a_study_outside_the_measured_cells_is_refused_before_any_record_is_consulted() {
    let looked = Artifact::to_bytes(
        &contract(categorical(true)),
        &ChoiceResultSpec::Posterior { look: 1 },
        "choice-looked",
        &ctx(),
    )
    .unwrap();
    let text = format!("{:?}", Artifact::measure_interval(&looked, &ctx()).unwrap_err());
    assert!(text.contains("protocol_not_measured"), "{text}");
    let with_source = Artifact::to_bytes(
        &contract(categorical(false)),
        &ChoiceResultSpec::Posterior { look: 0 },
        "choice-source",
        &ctx(),
    )
    .unwrap();
    let text = format!("{:?}", Artifact::measure_interval(&with_source, &ctx()).unwrap_err());
    assert!(text.contains("protocol_not_measured"), "{text}");
    let stopping = Artifact::to_bytes(
        &contract(categorical(true)),
        &ChoiceResultSpec::Stopping { look: 1 },
        "choice-stop",
        &ctx(),
    )
    .unwrap();
    let text = format!("{:?}", Artifact::measure_interval(&stopping, &ctx()).unwrap_err());
    assert!(text.contains("protocol_not_measured"), "{text}");
}

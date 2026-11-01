//! The choice-study analysis contract: completeness, consistency, identity, row-level evidence
//! accounting through the evidence ledger, and paradigm authorization through the procedure
//! matrix.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::adaptive_design::{
    AssignmentFamily, ChoiceDesignContract, ChoiceDesignDeclaration, ChoiceLook, ChoiceRespondent,
    InferenceParadigm, MonitoringStrategy, OutcomeStructure, ProcedureCell, RetainedChoiceSource,
    StoppingRuleDeclaration, StoppingRuleKind, StoppingSpec,
};

fn cell(
    outcome: OutcomeStructure,
    monitoring: MonitoringStrategy,
    stopping: StoppingRuleKind,
) -> ProcedureCell {
    ProcedureCell {
        assignment: AssignmentFamily::FixedRandomization,
        outcome,
        monitoring,
        stopping,
    }
}

fn labels() -> Vec<String> {
    ["a", "b", "c"].map(String::from).to_vec()
}

fn respondents(n: u32) -> Vec<ChoiceRespondent> {
    // One choice each, as a categorical-choice outcome requires.
    (0..n)
        .map(|i| {
            let mut counts = vec![0, 0, 0];
            counts[(i % 3) as usize] = 1;
            ChoiceRespondent { id: format!("r{i}"), counts }
        })
        .collect()
}

fn source(label: &str, ids: &[&str]) -> RetainedChoiceSource {
    RetainedChoiceSource {
        label: label.into(),
        option_labels: labels(),
        meaning: "chosen in the pricing task".into(),
        population: "customers".into(),
        counts: vec![ids.len() as f64, 0.0, 0.0],
        observation_ids: ids.iter().map(|s| (*s).to_owned()).collect(),
        discount: 0.5,
        transfer_rationale: None,
        posterior_proof: None,
    }
}

fn final_only() -> ChoiceDesignDeclaration {
    ChoiceDesignDeclaration {
        label: "wave".into(),
        cell: cell(
            OutcomeStructure::CategoricalChoice,
            MonitoringStrategy::FinalAnalysisOnly,
            StoppingRuleKind::NoStopping,
        ),
        option_labels: labels(),
        meaning: "chosen in the pricing task".into(),
        population: "customers".into(),
        respondents: respondents(6),
        initial_prior: vec![1.0; 3],
        retained_sources: Vec::new(),
        looks: Some(vec![ChoiceLook { respondents: 6 }]),
        stopping_rule: Some(StoppingRuleDeclaration {
            kind: StoppingRuleKind::NoStopping,
            spec: StoppingSpec::NoStopping,
            ended_by_rule: false,
        }),
    }
}

fn looked() -> ChoiceDesignDeclaration {
    let mut d = final_only();
    d.cell = cell(
        OutcomeStructure::CategoricalChoice,
        MonitoringStrategy::FixedInterimLooks,
        StoppingRuleKind::DecisionTheoretic,
    );
    d.looks = Some(vec![ChoiceLook { respondents: 3 }, ChoiceLook { respondents: 6 }]);
    d.stopping_rule = Some(StoppingRuleDeclaration {
        kind: StoppingRuleKind::DecisionTheoretic,
        spec: StoppingSpec::PosteriorExpectedLoss { loss_threshold: 0.05 },
        ended_by_rule: true,
    });
    d
}

fn details(d: ChoiceDesignDeclaration) -> Vec<String> {
    ChoiceDesignContract::new(d)
        .expect_err("refused")
        .refusals
        .iter()
        .map(|r| r.detail.clone())
        .collect()
}

#[test]
fn a_complete_declaration_is_admitted_with_a_stable_identity_that_moves_with_every_field() {
    let base = ChoiceDesignContract::new(final_only()).expect("complete");
    assert_eq!(base.identity(), ChoiceDesignContract::new(final_only()).unwrap().identity());
    assert_eq!(base.options(), 3);
    let mut changed = final_only();
    changed.respondents[0].counts = vec![0, 1, 0];
    assert_ne!(base.identity(), ChoiceDesignContract::new(changed).unwrap().identity());
    let mut changed = final_only();
    changed.initial_prior = vec![1.0, 1.0, 2.0];
    assert_ne!(base.identity(), ChoiceDesignContract::new(changed).unwrap().identity());
    let mut changed = final_only();
    changed.retained_sources = vec![source("old", &["old#0"])];
    assert_ne!(base.identity(), ChoiceDesignContract::new(changed).unwrap().identity());
}

#[test]
fn each_missing_or_inconsistent_element_is_its_own_refusal() {
    let mut d = final_only();
    d.looks = None;
    d.stopping_rule = None;
    assert_eq!(
        details(d),
        ["choice_inference.looks_missing", "choice_inference.stopping_rule_missing"]
    );
    let mut d = final_only();
    d.meaning = " ".into();
    assert_eq!(details(d), ["choice_inference.frame_incomplete"]);
    let mut d = final_only();
    d.option_labels = vec!["a".into(), "a".into(), "c".into()];
    assert!(details(d).contains(&"choice_inference.options_invalid".to_owned()));
    let mut d = final_only();
    d.respondents[1].id = "r0".into();
    assert_eq!(details(d), ["choice_inference.respondents_invalid"]);
    let mut d = final_only();
    d.looks = Some(vec![ChoiceLook { respondents: 5 }]);
    assert_eq!(details(d), ["choice_inference.looks_not_ordered"]);
    let mut d = looked();
    d.looks = Some(vec![ChoiceLook { respondents: 6 }]);
    assert_eq!(details(d), ["choice_inference.rule_mismatch"]);
    let mut d = looked();
    d.stopping_rule.as_mut().unwrap().spec =
        StoppingSpec::PosteriorProbabilityBest { threshold: 0.9 };
    assert_eq!(details(d), ["choice_inference.rule_mismatch"]);
    let mut d = looked();
    d.stopping_rule.as_mut().unwrap().spec =
        StoppingSpec::PosteriorExpectedLoss { loss_threshold: 1.5 };
    assert_eq!(details(d), ["choice_inference.rule_mismatch"]);
    let mut d = final_only();
    d.cell = cell(
        OutcomeStructure::SingleOutcome,
        MonitoringStrategy::FinalAnalysisOnly,
        StoppingRuleKind::NoStopping,
    );
    assert_eq!(details(d), ["choice_inference.cell_not_choice"]);
}

#[test]
fn heterogeneity_is_not_identified_without_repeated_choices() {
    let mut d = final_only();
    d.cell = cell(
        OutcomeStructure::RepeatedChoice,
        MonitoringStrategy::FinalAnalysisOnly,
        StoppingRuleKind::NoStopping,
    );
    d.respondents.iter_mut().for_each(|r| r.counts = vec![1, 0, 0]);
    assert_eq!(details(d), ["choice_inference.heterogeneity_not_identified"]);
}

#[test]
fn repeated_choosers_are_not_pooled_and_a_pooled_source_is_not_a_hierarchical_prior() {
    let mut d = final_only();
    d.respondents[0].counts = vec![2, 1, 1];
    assert_eq!(details(d), ["choice_inference.categorical_one_choice"]);
    let mut d = final_only();
    d.cell = cell(
        OutcomeStructure::RepeatedChoice,
        MonitoringStrategy::FinalAnalysisOnly,
        StoppingRuleKind::NoStopping,
    );
    d.respondents[0].counts = vec![2, 1, 1];
    assert!(ChoiceDesignContract::new(d.clone()).is_ok());
    d.retained_sources = vec![source("old", &["old#0"])];
    assert_eq!(details(d), ["choice_inference.hierarchical_source_unsupported"]);
}

#[test]
fn a_source_is_scoped_to_the_target_options_meaning_and_population() {
    let mut d = final_only();
    let mut s = source("old", &["old#0"]);
    s.option_labels = vec!["a".into(), "c".into(), "b".into()];
    d.retained_sources = vec![s];
    assert_eq!(details(d), ["choice_inference.source_parameterization_mismatch"]);
    let mut d = final_only();
    let mut s = source("old", &["old#0"]);
    s.meaning = "stated intention in a survey".into();
    d.retained_sources = vec![s];
    assert_eq!(details(d), ["choice_inference.source_meaning_mismatch"]);
    let mut d = final_only();
    let mut s = source("old", &["old#0"]);
    s.population = "prospects".into();
    d.retained_sources = vec![s.clone()];
    assert_eq!(details(d), ["choice_inference.source_population_mismatch"]);
    let mut d = final_only();
    s.transfer_rationale = Some("same market, same task".into());
    d.retained_sources = vec![s];
    assert_eq!(details(d), ["choice_inference.source_population_mismatch"]);
    let mut d = final_only();
    let mut s = source("old", &["old#0"]);
    s.transfer_rationale = Some("not needed".into());
    d.retained_sources = vec![s];
    assert_eq!(details(d), ["choice_inference.transfer_rationale_unneeded"]);
}

#[test]
fn the_ledger_refuses_sources_that_share_rows_with_the_respondents_or_each_other() {
    let mut d = final_only();
    d.retained_sources = vec![source("old", &["r2#0", "x"])];
    assert_eq!(details(d), ["choice_inference.source_overlaps_respondents"]);
    let mut d = final_only();
    d.retained_sources = vec![source("one", &["p", "q"]), source("two", &["q", "z"])];
    assert_eq!(details(d), ["choice_inference.sources_overlap"]);
    let mut d = final_only();
    d.retained_sources = vec![source("one", &["p", "q"]), source("two", &["z"])];
    let contract = ChoiceDesignContract::new(d).expect("disjoint rows");
    assert_eq!(contract.evidence().sources.len(), 2);
}

#[test]
fn the_matrix_authorizes_paradigms_for_the_cell() {
    let contract = ChoiceDesignContract::new(looked()).expect("complete");
    for paradigm in
        [InferenceParadigm::BayesianPosterior, InferenceParadigm::DecisionTheoreticStopping]
    {
        let granted = contract.admit_paradigm(paradigm).expect("licensed");
        assert_eq!(granted.contract_identity, contract.identity());
    }
    for refused in [
        InferenceParadigm::FrequentistFixedSample,
        InferenceParadigm::AnytimeValidFrequentist,
        InferenceParadigm::FrequentistGroupSequential,
    ] {
        assert!(contract.admit_paradigm(refused).is_err(), "{refused:?}");
    }
    let contract = ChoiceDesignContract::new(final_only()).expect("complete");
    assert!(contract.admit_paradigm(InferenceParadigm::DecisionTheoreticStopping).is_err());
    assert!(contract.admit_paradigm(InferenceParadigm::BayesianPosterior).is_ok());
}

#[test]
fn looks_expose_nested_respondent_prefixes() {
    let contract = ChoiceDesignContract::new(looked()).expect("complete");
    assert_eq!(contract.looks(), 2);
    assert_eq!(contract.respondents_at(0).unwrap().len(), 3);
    assert_eq!(contract.respondents_at(1).unwrap().len(), 6);
    assert_eq!(
        contract.respondents_at(2).unwrap_err().detail,
        "choice_inference.look_out_of_range"
    );
}

#[test]
fn source_counts_must_match_distinct_original_rows() {
    let mut d = final_only();
    let mut s = source("old", &["old#0", "old#1"]);
    s.counts = vec![10.0, 0.0, 0.0];
    d.retained_sources = vec![s];
    assert!(details(d).contains(&"choice_inference.source_invalid".to_owned()));

    let mut d = final_only();
    d.retained_sources = vec![source("old", &["duplicate", "duplicate"])];
    assert!(details(d).contains(&"choice_inference.source_invalid".to_owned()));

    let mut d = final_only();
    d.retained_sources = vec![source("old", &["person#0", "person#1"])];
    assert!(details(d).contains(&"choice_inference.source_invalid".to_owned()));
    let mut d = final_only();
    d.retained_sources = vec![source("old", &["r0"])];
    assert!(details(d).contains(&"choice_inference.source_overlaps_respondents".to_owned()));

    let mut d = final_only();
    let mut s = source("old", &["old#0"]);
    s.counts = vec![0.5, 0.5, 0.0];
    d.retained_sources = vec![s];
    assert!(details(d).contains(&"choice_inference.source_invalid".to_owned()));
}

#[test]
fn aggregate_rows_are_bounded_before_ledger_materialization() {
    let mut d = final_only();
    d.cell.outcome = OutcomeStructure::RepeatedChoice;
    d.respondents = (0..11)
        .map(|i| ChoiceRespondent { id: format!("r{i}"), counts: vec![100_000, 0, 0] })
        .collect();
    d.looks = Some(vec![ChoiceLook { respondents: 11 }]);
    assert!(details(d).contains(&"choice_inference.ledger_limit".to_owned()));
}

#[test]
fn withdrawing_a_source_removes_current_inference_authority() {
    let mut declaration = final_only();
    declaration.retained_sources = vec![source("old", &["old#0"])];
    let mut contract = ChoiceDesignContract::new(declaration).unwrap();
    assert!(contract.admit_paradigm(InferenceParadigm::BayesianPosterior).is_ok());
    contract.withdraw_retained_source("old", "source withdrawn").unwrap();
    let refusal = contract.admit_paradigm(InferenceParadigm::BayesianPosterior).unwrap_err();
    assert_eq!(refusal.code, "evidence_source_withdrawn");
}

//! Focused coverage for graphless randomized ITT routing.

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{RandomizedEffectQuery, Study};
use antecedent_core::VariableId;
use antecedent_data::TabularData;

#[test]
fn calibrated_randomized_intervals_round_trip_and_reject_tampering() {
    use antecedent_core::RandomizationDesign;

    fn run_case(query: RandomizedEffectQuery, outcomes: &[f64], design: &str) {
        let data = TabularData::from_f64_columns([("outcome", outcomes)]).unwrap();
        let ctx = ExecutionContext::for_tests(0xCA11_BA7E);
        let prepared = Study::tabular(data.clone()).query(query).build().unwrap()
            .prepare(&ctx).unwrap();
        let result = prepared.estimate(&data, &ctx).unwrap();
        let fit = result.randomized_effect.as_ref().unwrap();
        let [lower, upper] = fit.interval_95.expect("supported randomized design interval");
        assert_eq!(fit.assignment_design.as_ref(), design);
        assert_eq!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
        assert!(lower <= fit.effect && fit.effect <= upper);
        assert!(fit.standard_error.unwrap() > 0.0);
        assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::AnalyticSe);
        let executed_contract = prepared.contract_for_result(&result).unwrap();
        assert_eq!(executed_contract.support_status, result.support_status);
        let support = executed_contract.reasoning.support.as_ref().unwrap();
        assert_eq!(support.matrix_status.as_ref(), "licensed");
        assert!(support.matrix_coordinate.is_some());
        let bytes = prepared.encode_contracted_result(&result, "randomized-interval", &ctx).unwrap();
        let (encoded, header, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(artifact.randomized_effect.as_ref().unwrap().interval_95, Some([lower, upper]));
        assert_eq!(artifact.randomized_effect.as_ref().unwrap().graphless_support_status.as_deref(),
            Some("licensed"));
        let mut forged_license = artifact.clone();
        forged_license.randomized_effect.as_mut().unwrap().graphless_support_status =
            Some("refused".into());
        assert!(antecedent_io::encode_analysis_result_artifact(
            &forged_license, header.variable_names.clone(), "forged-license").is_err());
        if design == "factorial_2x2" {
            let mut missing_contrast = artifact.clone();
            missing_contrast.randomized_effect.as_mut().unwrap().factorial_interaction_interval_95 = None;
            assert!(antecedent_io::encode_analysis_result_artifact(
                &missing_contrast, header.variable_names.clone(), "missing-factorial-interval").is_err());
        }
        if design == "multi_arm" {
            let mut missing_action = artifact.clone();
            missing_action.randomized_effect.as_mut().unwrap().multi_arm_intervals_95[2] = None;
            assert!(antecedent_io::encode_analysis_result_artifact(
                &missing_action, header.variable_names.clone(), "missing-action-interval").is_err());
        }
        {
            // A pre-matrix artifact has no support-status field. It remains
            // readable as a legacy off-axis result, but a new encoder must
            // restamp the exact licensed status before issuing an artifact.
            let mut legacy_body = artifact.clone();
            legacy_body.randomized_effect.as_mut().unwrap().graphless_support_status = None;
            assert!(antecedent_io::encode_analysis_result_artifact(
                &legacy_body, header.variable_names.clone(), "legacy-new-encode").is_err());
            let mut legacy_container = encoded.clone();
            let body_index = legacy_container.sections.iter().position(|section|
                section.id == "analysis_result.body").unwrap();
            let body_bytes = antecedent_io::to_cbor(&legacy_body).unwrap();
            let (descriptor, section) = antecedent_io::pack_section_shared(
                "analysis_result.body", "application/cbor", body_bytes.into(),
                antecedent_io::CompressPolicy::Auto,
            );
            legacy_container.sections[body_index] = section;
            legacy_container.manifest.sections[body_index] = descriptor;
            let mut old_bytes = Vec::new();
            legacy_container.write_to(&mut old_bytes).unwrap();
            let (_, _, old_body) = antecedent_io::decode_analysis_result_artifact(&old_bytes).unwrap();
            assert!(old_body.randomized_effect.unwrap().graphless_support_status.is_none());
        }
        let mut tampered = artifact.clone();
        tampered.randomized_effect.as_mut().unwrap().interval_95.as_mut().unwrap()[1] += 0.5;
        assert!(antecedent_io::encode_analysis_result_artifact(
            &tampered, header.variable_names.clone(), "forged-interval").is_err());
        let mut tampered = artifact;
        tampered.standard_error = Some(0.1);
        assert!(antecedent_io::encode_analysis_result_artifact(
            &tampered, header.variable_names, "forged-se").is_err());
    }

    let units = |n: usize| (0..n).map(|i| Arc::<str>::from(format!("unit-{i}")))
        .collect::<Vec<_>>();
    let n = 60;
    let assignment = (0..n).map(|i| i < 30).collect::<Vec<_>>();
    let outcomes = (0..n).map(|i| 1.0 + 2.0 * f64::from(assignment[i])
        + 0.5 * (i as f64 * 0.3).sin()).collect::<Vec<_>>();
    run_case(RandomizedEffectQuery::with_design(
        RandomizationDesign::Complete { treated_units: 30 }, VariableId::from_raw(0),
        assignment, vec![0.5; n], units(n), units(n), ("control", "treated"),
    ), &outcomes, "complete");

    let n = 400;
    let assignment = (0..n).map(|i| i % 2 == 0).collect::<Vec<_>>();
    let outcomes = (0..n).map(|i| 1.0 + 2.0 * f64::from(assignment[i])
        + 0.5 * (i as f64 * 0.3).sin()).collect::<Vec<_>>();
    run_case(RandomizedEffectQuery::bernoulli_itt(
        VariableId::from_raw(0), assignment, vec![0.5; n], units(n), units(n),
        ("control", "treated"),
    ), &outcomes, "bernoulli");

    let arms = (0..n).map(|i| i % 4).collect::<Vec<_>>();
    let outcomes = (0..n).map(|i| 1.0 + arms[i] as f64
        + 0.5 * (i as f64 * 0.3).sin()).collect::<Vec<_>>();
    let fit = RandomizedEffectQuery::with_design(
        RandomizationDesign::MultiArm {
            assignment: Arc::from(arms.clone()), probabilities: vec![vec![0.25; 4]; n].into(),
            arms: ["control", "a", "b", "c"].map(Arc::<str>::from).into(),
        }, VariableId::from_raw(0), arms.iter().map(|arm| *arm != 0).collect::<Vec<_>>(),
        vec![0.25; n], units(n), units(n), ("control", "a"),
    );
    run_case(fit, &outcomes, "multi_arm");

    let n = 120;
    let assignment = (0..n).map(|i| i / 2 < 30).collect::<Vec<_>>();
    let outcomes = (0..n).map(|i| 1.0 + 2.0 * f64::from(assignment[i])
        + 0.5 * (i as f64 * 0.3).sin()).collect::<Vec<_>>();
    let clusters = (0..n).map(|i| Arc::<str>::from(format!("cluster-{}", i / 2)))
        .collect::<Vec<_>>();
    run_case(RandomizedEffectQuery::with_design(
        RandomizationDesign::Cluster { treated_clusters: 30 }, VariableId::from_raw(0),
        assignment, vec![0.5; n], clusters, units(n), ("control", "treated"),
    ), &outcomes, "cluster");

    let assignment = (0..n).map(|i| i % 30 < 15).collect::<Vec<_>>();
    let outcomes = (0..n).map(|i| 1.0 + 2.0 * f64::from(assignment[i])
        + 0.5 * (i as f64 * 0.3).sin()).collect::<Vec<_>>();
    let blocks = (0..n).map(|i| Arc::<str>::from(format!("block-{}", i / 30)))
        .collect::<Vec<_>>();
    run_case(RandomizedEffectQuery::with_design(
        RandomizationDesign::Stratified { blocks: Arc::from(blocks),
            treated_per_row: Arc::from([15; 120]) }, VariableId::from_raw(0),
        assignment, vec![0.5; n], units(n), units(n), ("control", "treated"),
    ), &outcomes, "stratified");

    let first = (0..n).map(|i| i / 30 % 2 == 1).collect::<Vec<_>>();
    let second = (0..n).map(|i| i / 30 >= 2).collect::<Vec<_>>();
    let outcomes = (0..n).map(|i| 1.0 + 2.0 * f64::from(first[i])
        + f64::from(second[i]) + 0.5 * f64::from(first[i] && second[i])
        + 0.5 * (i as f64 * 0.3).sin()).collect::<Vec<_>>();
    run_case(RandomizedEffectQuery::with_design(
        RandomizationDesign::Factorial2x2 { second_factor_assignment: Arc::from(second),
            cell_counts: [30; 4], second_factor_arms: ("b0".into(), "b1".into()) },
        VariableId::from_raw(0), first, vec![0.5; n], units(n), units(n),
        ("control", "treated"),
    ), &outcomes, "factorial_2x2");
}

#[test]
fn graphless_interval_license_refuses_zero_variance_despite_assignment_support() {
    use antecedent_core::RandomizationDesign;

    let n = 60;
    let assignments = (0..n).map(|i| i < 30).collect::<Vec<_>>();
    let units = (0..n).map(|i| Arc::<str>::from(format!("unit-{i}"))).collect::<Vec<_>>();
    let outcomes = (0..n).map(|i| if i < 30 { 3.0 } else { 1.0 }).collect::<Vec<_>>();
    for design in [
        RandomizationDesign::Complete { treated_units: 30 },
        RandomizationDesign::Cluster { treated_clusters: 30 },
    ] {
        let data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
        let query = RandomizedEffectQuery::with_design(
            design, VariableId::from_raw(0), assignments.clone(), vec![0.5; n],
            units.clone(), units.clone(), ("control", "treated"),
        );
        let ctx = ExecutionContext::for_tests(0xD364_E123);
        let result = Study::tabular(data).query(query).build().unwrap().run(&ctx).unwrap();
        assert_eq!(result.randomized_effect.as_ref().unwrap().interval_95, None);
        assert_eq!(result.support_status, None);
    }
}

#[test]
fn multi_arm_retained_study_reports_all_contrasts_and_refuses_missing_support() {
    let outcomes = [0.0, 2.0, 5.0, 0.0, 2.0, 5.0];
    let data = TabularData::from_f64_columns([("outcome", &outcomes[..])]).unwrap();
    let assignment: Arc<[usize]> = [0, 1, 2, 0, 1, 2].into();
    let query = RandomizedEffectQuery::with_design(
        antecedent_core::RandomizationDesign::MultiArm {
            assignment: assignment.clone(),
            probabilities: {
                let mut rows = vec![vec![1.0 / 3.0; 3]; 6];
                rows[0] = vec![0.5, 0.1, 0.4];
                rows.into()
            },
            arms: ["control", "low", "high"].map(Arc::<str>::from).into(),
        },
        VariableId::from_raw(0),
        assignment.iter().map(|&arm| arm != 0).collect::<Vec<_>>(),
        [0.5, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        (0..6).map(|i| Arc::<str>::from(format!("unit-{i}"))).collect::<Vec<_>>(),
        (0..6).map(|i| Arc::<str>::from(format!("row-{i}"))).collect::<Vec<_>>(),
        ("control", "low"),
    );
    let ctx = ExecutionContext::for_tests(73);
    let prepared = Study::tabular(data.clone()).query(query.clone()).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let fit = result.randomized_effect.as_ref().unwrap();
    assert_eq!(fit.effect, 2.0);
    assert_eq!(fit.estimand.as_ref(), "multi_arm_itt");
    assert_eq!(fit.assignment_design.as_ref(), "multi_arm");
    assert!((fit.minimum_assignment_probability - 0.1).abs() < 1e-12);
    assert_eq!(fit.uncertainty.as_ref(), "multi_arm_covariance_free_variance_bound_no_interval");
    assert_eq!(fit.multi_arm_values.iter().map(|(_, mean, _, _)| *mean).collect::<Vec<_>>(), [0.0, 2.0, 5.0]);
    assert_eq!(fit.multi_arm_values.iter().map(|(_, _, _, support)| *support).collect::<Vec<_>>(), [2, 2, 2]);
    assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::None);
    assert_eq!(result.support_status, None);
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, antecedent_core::Assumption::Custom { id, .. } if id.as_ref() == "known_random_assignment"
    )));
    assert_eq!(prepared.estimate(&data, &ctx).unwrap().randomized_effect.as_ref().unwrap(), fit);
    let bytes = prepared.encode_contracted_result(&result, "multi-arm", &ctx).unwrap();
    let (_, header, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(artifact.randomized_effect.as_ref().unwrap().multi_arm_values[2].1, 5.0);
    assert!((artifact.randomized_effect.as_ref().unwrap().minimum_assignment_probability - 0.1).abs() < 1e-12);
    let mut wrong_support = artifact.clone();
    wrong_support.randomized_effect.as_mut().unwrap().minimum_assignment_probability = 1.0 / 3.0;
    assert!(antecedent_io::encode_analysis_result_artifact(&wrong_support, header.variable_names.clone(), "wrong-support").is_err());
    let mut wrong_arm = artifact.clone();
    // The serialized result must retain a coherent primary contrast.
    wrong_arm.randomized_effect.as_mut().unwrap().multi_arm_values[1].1 = 6.0;
    assert!(antecedent_io::encode_analysis_result_artifact(&wrong_arm, header.variable_names.clone(), "wrong-arm").is_err());
    let mut fabricated = artifact;
    fabricated.standard_error = Some(0.5);
    assert!(antecedent_io::encode_analysis_result_artifact(&fabricated, header.variable_names, "fabricated-multi-arm").is_err());
    let mut missing = query;
    missing.design = antecedent_core::RandomizationDesign::MultiArm {
        assignment: [0, 1, 1, 0, 1, 1].into(),
        probabilities: vec![vec![1.0 / 3.0; 3]; 6].into(),
        arms: ["control", "low", "high"].map(Arc::<str>::from).into(),
    };
    assert!(missing.validate().is_err());
}

#[test]
fn graphless_bernoulli_itt_runs_and_retains_design_units() {
    let outcomes = [3.0, 0.0, 4.0, 1.0];
    let data = TabularData::from_f64_columns([("outcome", &outcomes[..])]).unwrap();
    let query = RandomizedEffectQuery::bernoulli_itt(
        VariableId::from_raw(0),
        [true, false, true, false],
        [0.5; 4],
        ["a", "b", "c", "d"].map(Arc::<str>::from),
        ["r0", "r1", "r2", "r3"].map(Arc::<str>::from),
        ("control", "treated"),
    );
    let study = Study::tabular(data.clone()).query(query.clone()).build().unwrap();
    let ctx = ExecutionContext::for_tests(3);
    let result = study.run(&ctx).unwrap();

    let estimate = result.randomized_effect.as_ref().unwrap();
    assert_eq!(estimate.effect, 3.0);
    assert_eq!(estimate.variance_upper_bound, 3.25);
    assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::None);
    assert_eq!(
        estimate.assignment_units.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
        ["a", "b", "c", "d"]
    );
    assert_eq!(
        estimate.outcome_units.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
        ["r0", "r1", "r2", "r3"]
    );
    assert_eq!(result.treatment, None);
    assert_eq!(result.support_status, None);

    let prepared =
        Study::tabular(data.clone()).query(query).build().unwrap().prepare(&ctx).unwrap();
    let refreshed = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(refreshed.randomized_effect.as_ref().unwrap().effect, 3.0);
    assert_eq!(refreshed.treatment, None);
}

#[test]
fn fixed_cuped_bernoulli_itt_adjusts_in_native_retained_execution() {
    let baseline = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
    let outcomes = [2.0, 0.0, 6.0, 4.0, 10.0, 8.0, 14.0, 12.0];
    let data = TabularData::from_f64_columns([
        ("outcome", &outcomes[..]), ("baseline", &baseline[..]),
    ]).unwrap();
    let query = RandomizedEffectQuery::bernoulli_itt(
        VariableId::from_raw(0),
        [true, false, true, false, true, false, true, false],
        [0.5; 8],
        (0..8).map(|i| Arc::<str>::from(format!("u{i}"))).collect::<Vec<_>>(),
        (0..8).map(|i| Arc::<str>::from(format!("y{i}"))).collect::<Vec<_>>(),
        ("control", "treated"),
    ).with_fixed_cuped(VariableId::from_raw(1), 4.0);
    let result = Study::tabular(data).query(query).build().unwrap()
        .run(&ExecutionContext::for_tests(47)).unwrap();
    let adjusted = result.randomized_effect.as_ref().unwrap();
    assert_eq!(adjusted.effect, 2.0);
    assert_eq!(adjusted.variance_upper_bound, 1.0);
    assert_eq!(adjusted.uncertainty.as_ref(), "bernoulli_fixed_cuped_ht_conservative_variance_no_interval");
    assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::None);
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, antecedent_core::Assumption::Custom { id, .. }
        if id.as_ref() == "fixed_pre_assignment_cuped"
    )));
}

#[test]
fn multi_covariate_ancova_runs_in_retained_study_and_artifact() {
    let assignment = [false, true, false, true, true, false, true, false];
    let x1 = [0., 1., 2., 3., 4., 5., 6., 7.];
    let x2 = [1., 0., 1., 0., 1., 0., 1., 0.];
    let y = (0..8).map(|i| 3. + 2. * f64::from(assignment[i]) + 4. * x1[i] - x2[i]).collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([
        ("outcome", y.as_slice()), ("baseline_a", &x1), ("baseline_b", &x2),
    ]).unwrap();
    let base = RandomizedEffectQuery::bernoulli_itt(
        VariableId::from_raw(0), assignment, [0.5; 8],
        (0..8).map(|i| Arc::<str>::from(format!("u{i}"))).collect::<Vec<_>>(),
        (0..8).map(|i| Arc::<str>::from(format!("y{i}"))).collect::<Vec<_>>(),
        ("control", "treated"),
    );
    let query = base.clone().with_ancova(vec![VariableId::from_raw(1), VariableId::from_raw(2)]);
    let ctx = ExecutionContext::for_tests(67);
    let prepared = Study::tabular(data.clone()).query(query).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let adjusted = result.randomized_effect.as_ref().unwrap();
    assert!((adjusted.effect - 2.).abs() < 1e-10);
    assert!(adjusted.variance_upper_bound < 1e-20);
    assert_eq!(adjusted.uncertainty.as_ref(), "bernoulli_ancova_hc0_variance_no_interval");
    assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::None);
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, antecedent_core::Assumption::Custom { id, .. }
        if id.as_ref() == "ancova_pre_assignment_covariates"
    )));
    let bytes = prepared.encode_contracted_result(&result, "ancova", &ctx).unwrap();
    let (_, header, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(artifact.randomized_effect.as_ref().unwrap().uncertainty, "bernoulli_ancova_hc0_variance_no_interval");
    let mut forged = artifact;
    forged.standard_error = Some(0.1);
    assert!(antecedent_io::encode_analysis_result_artifact(&forged, header.variable_names, "forged-ancova").is_err());
    assert!(base.clone().with_ancova(vec![VariableId::from_raw(1), VariableId::from_raw(1)]).validate().is_err());
    assert!(base.with_fixed_cuped(VariableId::from_raw(1), 4.).with_ancova(vec![VariableId::from_raw(2)]).validate().is_err());
}

#[test]
fn bernoulli_encouragement_retains_wald_cace_and_refuses_fabricated_interval() {
    let outcomes = [5.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let data = TabularData::from_f64_columns([("outcome", &outcomes[..])]).unwrap();
    let base = RandomizedEffectQuery::bernoulli_itt(
        VariableId::from_raw(0),
        [true, false, true, false, true, false, true, false],
        [0.5; 8],
        (0..8).map(|i| Arc::<str>::from(format!("u{i}"))).collect::<Vec<_>>(),
        (0..8).map(|i| Arc::<str>::from(format!("y{i}"))).collect::<Vec<_>>(),
        ("control", "encouraged"),
    );
    let query = base.clone().with_received_treatment([true, false, true, false, false, false, false, false]);
    let ctx = ExecutionContext::for_tests(57);
    let prepared = Study::tabular(data.clone()).query(query).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let effect = result.randomized_effect.as_ref().unwrap();
    assert_eq!(effect.effect, 4.0);
    assert_eq!(effect.intention_to_treat_effect, Some(2.0));
    assert_eq!(effect.first_stage_effect, Some(0.5));
    assert!((effect.variance_upper_bound - 16.0 / 7.0).abs() < 1e-12);
    assert_eq!(effect.estimand.as_ref(), "cace_late");
    assert_eq!(effect.uncertainty.as_ref(), "bernoulli_wald_cace_influence_variance_no_interval");
    assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::None);
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, antecedent_core::Assumption::Custom { id, .. }
        if id.as_ref() == "exclusion_restriction"
    )));
    let bytes = prepared.encode_contracted_result(&result, "cace", &ctx).unwrap();
    let (_, header, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(artifact.randomized_effect.as_ref().unwrap().first_stage_effect, Some(0.5));
    let mut fabricated = artifact;
    fabricated.standard_error = Some(1.0);
    assert!(antecedent_io::encode_analysis_result_artifact(&fabricated, header.variable_names, "fabricated-cace").is_err());
    let zero_stage = base.with_received_treatment([false; 8]);
    assert!(Study::tabular(data).query(zero_stage).build().unwrap().run(&ctx).is_err());
}

#[test]
fn one_sided_treatment_on_treated_is_distinct_and_refuses_two_sided_receipt() {
    // Only encouraged recipients have a +4 treatment response. Under one-sided
    // noncompliance and exclusion, the recipient ATT equals the Wald ratio.
    let outcomes = [5.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let data = TabularData::from_f64_columns([("outcome", &outcomes[..])]).unwrap();
    let base = RandomizedEffectQuery::bernoulli_itt(
        VariableId::from_raw(0),
        [true, false, true, false, true, false, true, false],
        [0.5; 8],
        (0..8).map(|i| Arc::<str>::from(format!("u{i}"))).collect::<Vec<_>>(),
        (0..8).map(|i| Arc::<str>::from(format!("y{i}"))).collect::<Vec<_>>(),
        ("control", "encouraged"),
    );
    let query = base.clone().with_treatment_on_treated([true, false, true, false, false, false, false, false]);
    let ctx = ExecutionContext::for_tests(63);
    let prepared = Study::tabular(data.clone()).query(query).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let section = result.randomized_effect.as_ref().unwrap();
    assert_eq!(section.estimand.as_ref(), "treatment_on_treated");
    assert_eq!(section.effect, 4.0);
    assert_eq!(section.intention_to_treat_effect, Some(2.0));
    assert_eq!(section.first_stage_effect, Some(0.5));
    assert!(section.interval_95.is_none());
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, antecedent_core::Assumption::Custom { id, .. }
        if id.as_ref() == "one_sided_noncompliance"
    )));
    let bytes = prepared.encode_contracted_result(&result, "tot", &ctx).unwrap();
    let (_, header, mut artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(artifact.randomized_effect.as_ref().unwrap().estimand, "treatment_on_treated");
    artifact.randomized_effect.as_mut().unwrap().estimand = "cace_late".into();
    assert!(antecedent_io::encode_analysis_result_artifact(&artifact, header.variable_names, "forged-tot").is_err());
    let two_sided = base.with_treatment_on_treated([true, true, true, false, false, false, false, false]);
    assert!(two_sided.validate().is_err());
}

#[test]
fn complete_randomization_enumerates_fisher_sharp_null_without_interval() {
    let outcomes = [1.0, 2.0, 3.0, 4.0];
    let data = TabularData::from_f64_columns([("outcome", &outcomes[..])]).unwrap();
    let base = RandomizedEffectQuery::with_design(
        antecedent_core::RandomizationDesign::Complete { treated_units: 2 },
        VariableId::from_raw(0), [true, true, false, false], [0.5; 4],
        ["u0", "u1", "u2", "u3"].map(Arc::<str>::from),
        ["y0", "y1", "y2", "y3"].map(Arc::<str>::from),
        ("control", "treated"),
    );
    let ctx = ExecutionContext::for_tests(61);
    let prepared = Study::tabular(data.clone()).query(base.clone().with_exact_randomization_test())
        .build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let effect = result.randomized_effect.as_ref().unwrap();
    assert_eq!(effect.effect, -2.0);
    assert_eq!(effect.randomization_allocations, Some(6));
    assert!((effect.randomization_p_value.unwrap() - 1.0 / 3.0).abs() < 1e-12);
    assert_eq!(prepared.estimate(&data, &ctx).unwrap().randomized_effect.as_ref().unwrap().randomization_p_value,
        effect.randomization_p_value);
    assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::None);
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, antecedent_core::Assumption::Custom { id, .. }
        if id.as_ref() == "fisher_sharp_null_two_sided"
    )));
    let bytes = prepared.encode_contracted_result(&result, "fisher", &ctx).unwrap();
    let (_, header, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(artifact.randomized_effect.as_ref().unwrap().randomization_allocations, Some(6));
    let mut fabricated = artifact;
    fabricated.randomized_effect.as_mut().unwrap().randomization_p_value = Some(0.0);
    assert!(antecedent_io::encode_analysis_result_artifact(&fabricated, header.variable_names, "fabricated-fisher").is_err());
    let bernoulli = RandomizedEffectQuery::bernoulli_itt(
        VariableId::from_raw(0), [true, false, true, false], [0.5; 4],
        ["u0", "u1", "u2", "u3"].map(Arc::<str>::from),
        ["y0", "y1", "y2", "y3"].map(Arc::<str>::from),
        ("control", "treated"),
    ).with_exact_randomization_test();
    assert!(bernoulli.validate().is_err());
}

#[test]
fn fixed_cell_factorial_reports_both_main_effects_and_interaction_without_interval() {
    let outcomes = [0.0, 2.0, 2.0, 4.0, 1.0, 3.0, 5.0, 7.0];
    let data = TabularData::from_f64_columns([("outcome", &outcomes[..])]).unwrap();
    let second = [false, false, false, false, true, true, true, true];
    let primary = [false, false, true, true, false, false, true, true];
    let query = RandomizedEffectQuery::with_design(
        antecedent_core::RandomizationDesign::Factorial2x2 {
            second_factor_assignment: second.into(),
            cell_counts: [2; 4],
            second_factor_arms: (Arc::from("off"), Arc::from("on")),
        },
        VariableId::from_raw(0), primary, [0.5; 8],
        (0..8).map(|i| Arc::<str>::from(format!("u{i}"))).collect::<Vec<_>>(),
        (0..8).map(|i| Arc::<str>::from(format!("y{i}"))).collect::<Vec<_>>(),
        ("control", "treated"),
    );
    let ctx = ExecutionContext::for_tests(67);
    let prepared = Study::tabular(data.clone()).query(query.clone()).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let factorial = result.randomized_effect.as_ref().unwrap();
    assert_eq!(factorial.effect, 3.0);
    assert_eq!(factorial.second_factor_effect, Some(2.0));
    assert_eq!(factorial.factorial_interaction, Some(2.0));
    assert_eq!(factorial.variance_upper_bound, 1.0);
    assert_eq!(factorial.second_factor_variance, Some(1.0));
    assert_eq!(factorial.factorial_interaction_variance, Some(4.0));
    assert_eq!(factorial.estimand.as_ref(), "factorial_primary_main_effect");
    assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::None);
    assert_eq!(result.support_status, None);
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, antecedent_core::Assumption::Custom { id, .. }
        if id.as_ref() == "known_random_assignment"
    )));
    let bytes = prepared.encode_contracted_result(&result, "factorial", &ctx).unwrap();
    let (_, header, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(artifact.randomized_effect.as_ref().unwrap().factorial_interaction, Some(2.0));
    let mut fabricated = artifact;
    fabricated.randomized_effect.as_mut().unwrap().factorial_interaction_variance = Some(0.0);
    assert!(antecedent_io::encode_analysis_result_artifact(&fabricated, header.variable_names, "fabricated-factorial").is_err());
    let mut unsupported = query;
    unsupported.assignment_probabilities = [0.25; 8].into();
    assert!(unsupported.validate().is_err());
}

#[test]
fn switchback_itt_retains_periods_and_sequence_variance() {
    let assignment = [true, false, false, true].repeat(4);
    let outcomes = (0..4).flat_map(|sequence| {
        [true, false, false, true].into_iter().map(move |treated| {
            10.0 * sequence as f64 + if treated { sequence as f64 + 1.0 } else { 0.0 }
        })
    }).collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
    let sequences = (0..4).flat_map(|sequence| {
        (0..4).map(move |_| Arc::<str>::from(format!("s{sequence}")))
    }).collect::<Vec<_>>();
    let periods = (0..4).flat_map(|_| {
        (0..4).map(|period| Arc::<str>::from(format!("p{period}")))
    }).collect::<Vec<_>>();
    let query = RandomizedEffectQuery::with_design(
        antecedent_core::RandomizationDesign::Switchback { periods: periods.clone().into() },
        VariableId::from_raw(0), assignment, [0.5; 16], sequences,
        (0..16).map(|i| Arc::<str>::from(format!("row-{i}"))).collect::<Vec<_>>(),
        ("off", "on"),
    );
    let context = ExecutionContext::for_tests(48);
    let prepared = Study::tabular(data.clone()).query(query).build().unwrap()
        .prepare(&context).unwrap();
    let result = prepared.estimate(&data, &context).unwrap();
    let estimate = result.randomized_effect.as_ref().unwrap();
    assert_eq!(estimate.effect, 2.5);
    assert!((estimate.variance_upper_bound - 5.0 / 12.0).abs() < 1e-12);
    assert_eq!(estimate.assignment_design.as_ref(), "switchback");
    assert_eq!(estimate.periods.as_ref(), periods.as_slice());
    assert_eq!(estimate.uncertainty.as_ref(), "switchback_independent_sequence_sandwich_variance_no_interval");
    assert_eq!(result.interval.as_ref().unwrap().method, antecedent_core::IntervalMethod::None);
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, antecedent_core::Assumption::Custom { id, .. }
        if id.as_ref() == "switchback_no_carryover"
    )));
    let bytes = prepared.encode_contracted_result(&result, "switchback", &context).unwrap();
    let (_, header, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(artifact.estimate, Some(2.5));
    assert_eq!(artifact.standard_error, None);
    assert_eq!(artifact.interval_lower, None);
    assert_eq!(artifact.interval_upper, None);
    assert!((artifact.randomized_effect.as_ref().unwrap().variance - 5.0 / 12.0).abs() < 1e-12);
    assert_eq!(artifact.randomized_effect.as_ref().unwrap().periods,
        ["p0", "p1", "p2", "p3"].repeat(4));
    let mut fabricated = artifact.clone();
    fabricated.standard_error = Some(0.1);
    assert!(antecedent_io::encode_analysis_result_artifact(
        &fabricated, header.variable_names, "fabricated-switchback"
    ).is_err());
    let antecedent_io::CausalQueryWire::RandomizedEffect(wire) = artifact.query else {
        panic!("switchback artifact must retain randomized query");
    };
    assert_eq!(wire.periods, ["p0", "p1", "p2", "p3"].repeat(4));
}

#[test]
fn graphless_complete_and_stratified_itt_use_neyman_variance() {
    let context = ExecutionContext::for_tests(4);
    let outcomes = [2.0, 0.0, 4.0, 2.0, 6.0, 4.0, 8.0, 6.0];
    let data = TabularData::from_f64_columns([("outcome", &outcomes[..])]).unwrap();
    let units = (0..8).map(|i| Arc::<str>::from(format!("unit-{i}"))).collect::<Vec<_>>();
    let assignment = [true, false, true, false, true, false, true, false];

    let complete = RandomizedEffectQuery::with_design(
        antecedent_core::RandomizationDesign::Complete { treated_units: 4 },
        VariableId::from_raw(0),
        assignment,
        [0.5; 8],
        units.clone(),
        units.clone(),
        ("control", "treated"),
    );
    let complete_result =
        Study::tabular(data.clone()).query(complete).build().unwrap().run(&context).unwrap();
    let complete_effect = complete_result.randomized_effect.unwrap();
    assert_eq!(complete_effect.effect, 2.0);
    assert!((complete_effect.variance_upper_bound - 10.0 / 3.0).abs() < 1e-12);
    assert_eq!(complete_effect.assignment_design.as_ref(), "complete");
    assert_eq!(complete_effect.control_units, 4);
    assert_eq!(complete_effect.treatment_units, 4);
    assert_eq!(
        complete_effect.uncertainty.as_ref(),
        "complete_neyman_variance_upper_bound_no_interval"
    );

    let blocks = ["north", "north", "north", "north", "south", "south", "south", "south"]
        .map(Arc::<str>::from);
    let stratified = RandomizedEffectQuery::with_design(
        antecedent_core::RandomizationDesign::Stratified {
            blocks: Arc::from(blocks),
            treated_per_row: Arc::from([2; 8]),
        },
        VariableId::from_raw(0),
        assignment,
        [0.5; 8],
        units.clone(),
        units,
        ("control", "treated"),
    );
    let stratified_study = Study::tabular(data.clone()).query(stratified).build().unwrap();
    let prepared = stratified_study.prepare(&context).unwrap();
    let stratified_effect = prepared.estimate(&data, &context).unwrap().randomized_effect.unwrap();
    assert_eq!(stratified_effect.effect, 2.0);
    assert!((stratified_effect.variance_upper_bound - 1.0).abs() < 1e-12);
    assert_eq!(stratified_effect.assignment_design.as_ref(), "stratified");
    assert_eq!(stratified_effect.blocks.len(), 8);
    assert_eq!(
        stratified_effect.uncertainty.as_ref(),
        "stratified_neyman_variance_upper_bound_no_interval"
    );
}

#[test]
fn cluster_randomized_itt_uses_cluster_totals_and_assignment_units() {
    let outcomes = [6.0, 8.0, 4.0, 1.0, 3.0, 2.0];
    let data = TabularData::from_f64_columns([("outcome", &outcomes[..])]).unwrap();
    let query = RandomizedEffectQuery::with_design(
        antecedent_core::RandomizationDesign::Cluster { treated_clusters: 2 },
        VariableId::from_raw(0),
        [true, true, true, false, false, false],
        [0.5; 6],
        ["a", "a", "b", "c", "c", "d"].map(Arc::<str>::from),
        ["r0", "r1", "r2", "r3", "r4", "r5"].map(Arc::<str>::from),
        ("control", "treated"),
    );
    let context = ExecutionContext::for_tests(37);
    let result =
        Study::tabular(data.clone()).query(query.clone()).build().unwrap().run(&context).unwrap();
    let estimate = result.randomized_effect.as_ref().unwrap();
    assert_eq!(estimate.effect, 4.0);
    assert!((estimate.variance_upper_bound - 104.0 / 9.0).abs() < 1e-12);
    assert_eq!(estimate.control_units, 2);
    assert_eq!(estimate.treatment_units, 2);
    assert_eq!(estimate.assignment_design.as_ref(), "cluster");
    assert_eq!(estimate.uncertainty.as_ref(), "cluster_neyman_variance_upper_bound_no_interval");
    assert!(result.identification.required_assumptions.entries.iter().any(|record| matches!(
        &record.assumption, antecedent_core::Assumption::Custom { id, .. }
        if id.as_ref() == "no_between_cluster_interference"
    )));
    assert_eq!(result.support_status, None);
    let prepared =
        Study::tabular(data.clone()).query(query).build().unwrap().prepare(&context).unwrap();
    assert_eq!(
        prepared.estimate(&data, &context).unwrap().randomized_effect,
        result.randomized_effect
    );
}

#[test]
fn cluster_randomized_itt_refuses_inconsistent_assignment_or_sparse_arms() {
    let outcomes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data = TabularData::from_f64_columns([("outcome", &outcomes[..])]).unwrap();
    let make = |assignment: Vec<bool>, treated_clusters: usize| {
        RandomizedEffectQuery::with_design(
            antecedent_core::RandomizationDesign::Cluster { treated_clusters },
            VariableId::from_raw(0),
            assignment,
            [0.5; 6],
            ["a", "a", "b", "c", "c", "d"].map(Arc::<str>::from),
            ["r0", "r1", "r2", "r3", "r4", "r5"].map(Arc::<str>::from),
            ("control", "treated"),
        )
    };
    let context = ExecutionContext::for_tests(38);
    assert!(
        Study::tabular(data.clone())
            .query(make(vec![true, false, true, false, false, false], 2))
            .build()
            .unwrap()
            .run(&context)
            .is_err()
    );
    assert!(
        Study::tabular(data)
            .query(make(vec![true, true, false, false, false, false], 1))
            .build()
            .unwrap()
            .run(&context)
            .is_err()
    );
}

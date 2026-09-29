//! Focused coverage for the graphless complier-effect (CACE/LATE and one-sided
//! treatment-on-treated) interval license and its retained artifact route.
#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "integration test asserts exact deterministic estimates and exercises every forgery gate"
)]

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{RandomizedEffectQuery, Study};
use antecedent_core::VariableId;
use antecedent_data::TabularData;

#[test]
fn licensed_complier_and_tot_intervals_round_trip_and_refuse_forgery() {
    fn run_case(query: RandomizedEffectQuery, outcomes: &[f64], design: &str, estimand: &str) {
        let data = TabularData::from_f64_columns([("outcome", outcomes)]).unwrap();
        let ctx = ExecutionContext::for_tests(0xC0FF_EE42);
        let prepared =
            Study::tabular(data.clone()).query(query).build().unwrap().prepare(&ctx).unwrap();
        let result = prepared.estimate(&data, &ctx).unwrap();
        let fit = result.randomized_effect.as_ref().unwrap();
        let [lower, upper] = fit.interval_95.expect("supported complier interval");
        assert_eq!(fit.assignment_design.as_ref(), "bernoulli");
        assert_eq!(fit.estimand.as_ref(), estimand);
        assert_eq!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
        assert!(lower <= fit.effect && fit.effect <= upper);
        assert!(fit.standard_error.unwrap() > 0.0);
        // The published interval is exactly point +/- z_{0.975} * SE.
        let z95 = 1.959_963_984_540_054;
        let se = fit.standard_error.unwrap();
        assert!((lower - (fit.effect - z95 * se)).abs() <= 1e-9);
        assert!((upper - (fit.effect + z95 * se)).abs() <= 1e-9);
        // The Wald ratio reconciles with its published components.
        assert!(
            (fit.effect - fit.intention_to_treat_effect.unwrap() / fit.first_stage_effect.unwrap())
                .abs()
                <= 1e-10
        );

        let executed_contract = prepared.contract_for_result(&result).unwrap();
        assert_eq!(executed_contract.support_status, result.support_status);
        let support = executed_contract.reasoning.support.as_ref().unwrap();
        assert_eq!(support.matrix_status.as_ref(), "licensed");
        assert_eq!(support.matrix_coordinate.as_deref(),
            Some(format!("graphless:complier_effect/{design}/wald_ratio_influence/pointwise_95_normal_interval").as_str()));

        let bytes = prepared.encode_contracted_result(&result, "complier-interval", &ctx).unwrap();
        let (_, header, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(artifact.randomized_effect.as_ref().unwrap().interval_95, Some([lower, upper]));
        assert_eq!(
            artifact.randomized_effect.as_ref().unwrap().graphless_support_status.as_deref(),
            Some("licensed")
        );

        // A forged support status is rejected by the artifact validator.
        let mut forged_license = artifact.clone();
        forged_license.randomized_effect.as_mut().unwrap().graphless_support_status =
            Some("refused".into());
        assert!(
            antecedent_io::encode_analysis_result_artifact(
                &forged_license,
                header.variable_names.clone(),
                "forged-license"
            )
            .is_err()
        );

        // A dropped status must be restamped by a new encoder.
        let mut legacy_body = artifact.clone();
        legacy_body.randomized_effect.as_mut().unwrap().graphless_support_status = None;
        assert!(
            antecedent_io::encode_analysis_result_artifact(
                &legacy_body,
                header.variable_names.clone(),
                "legacy-new-encode"
            )
            .is_err()
        );

        // A widened interval no longer matches point +/- z * SE.
        let mut tampered = artifact.clone();
        tampered.randomized_effect.as_mut().unwrap().interval_95.as_mut().unwrap()[1] += 0.5;
        assert!(
            antecedent_io::encode_analysis_result_artifact(
                &tampered,
                header.variable_names.clone(),
                "forged-interval"
            )
            .is_err()
        );

        // A fabricated top-level standard error is rejected.
        let mut tampered = artifact;
        tampered.standard_error = Some(0.1);
        assert!(
            antecedent_io::encode_analysis_result_artifact(
                &tampered,
                header.variable_names,
                "forged-se"
            )
            .is_err()
        );
    }

    let units =
        |n: usize| (0..n).map(|i| Arc::<str>::from(format!("unit-{i}"))).collect::<Vec<_>>();
    let n = 400;
    let assignment = (0..n).map(|i| i % 2 == 0).collect::<Vec<_>>();

    // Two-sided noncompliance: compliers take treatment iff encouraged, with a
    // few always-takers and never-takers so the first stage is below one.
    let cace_received =
        (0..n).map(|i| if assignment[i] { i % 5 != 0 } else { i % 7 == 0 }).collect::<Vec<_>>();
    let cace_outcomes = (0..n)
        .map(|i| 1.0 + 2.0 * f64::from(cace_received[i]) + 0.5 * (i as f64 * 0.3).sin())
        .collect::<Vec<_>>();
    run_case(
        RandomizedEffectQuery::bernoulli_itt(
            VariableId::from_raw(0),
            assignment.clone(),
            vec![0.5; n],
            units(n),
            units(n),
            ("control", "treated"),
        )
        .with_received_treatment(cace_received),
        &cace_outcomes,
        "bernoulli",
        "cace_late",
    );

    // One-sided noncompliance: no control-assigned unit receives treatment.
    let tot_received = (0..n).map(|i| assignment[i] && i % 5 != 0).collect::<Vec<_>>();
    let tot_outcomes = (0..n)
        .map(|i| 1.0 + 2.0 * f64::from(tot_received[i]) + 0.5 * (i as f64 * 0.3).sin())
        .collect::<Vec<_>>();
    run_case(
        RandomizedEffectQuery::bernoulli_itt(
            VariableId::from_raw(0),
            assignment,
            vec![0.5; n],
            units(n),
            units(n),
            ("control", "treated"),
        )
        .with_treatment_on_treated(tot_received),
        &tot_outcomes,
        "bernoulli_one_sided",
        "treatment_on_treated",
    );
}

#[test]
fn complier_interval_withheld_below_support_threshold() {
    let n = 100;
    let assignment = (0..n).map(|i| i % 2 == 0).collect::<Vec<_>>();
    let received = (0..n).map(|i| assignment[i] && i % 5 != 0).collect::<Vec<_>>();
    let outcomes = (0..n)
        .map(|i| 1.0 + 2.0 * f64::from(received[i]) + 0.5 * (i as f64 * 0.3).sin())
        .collect::<Vec<_>>();
    let units = (0..n).map(|i| Arc::<str>::from(format!("unit-{i}"))).collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
    let query = RandomizedEffectQuery::bernoulli_itt(
        VariableId::from_raw(0),
        assignment,
        vec![0.5; n],
        units.clone(),
        units,
        ("control", "treated"),
    )
    .with_received_treatment(received);
    let ctx = ExecutionContext::for_tests(0x51CE_1234);
    let result = Study::tabular(data).query(query).build().unwrap().run(&ctx).unwrap();
    assert_eq!(result.randomized_effect.as_ref().unwrap().interval_95, None);
    assert_eq!(result.support_status, None);
}

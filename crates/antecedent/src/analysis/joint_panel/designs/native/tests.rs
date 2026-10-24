// SPDX-License-Identifier: MIT OR Apache-2.0
use super::*;
use antecedent_estimate::joint_design::{instrumented::*, rollout::*, switchback::*, synthetic::*};
fn context() -> ExecutionContext {
    ExecutionContext::for_tests(445)
}
fn replay(prepared: &PreparedNativeJointDesign, inference: JointDesignInference) {
    let result = prepared.execute(inference, &context()).unwrap();
    let bytes = result.export_artifact().unwrap();
    let numerical = replay_native_joint_design(&bytes, &context()).unwrap();
    assert_eq!(numerical, *result.law());
    assert_eq!(prepared.consume_current(&bytes, &context()).unwrap().law(), result.law());
}
#[test]
fn native_history_compiler_replays_whole_schedule_design_and_actual_gaussian_likelihood() {
    let schedules = (0..4).flat_map(|a| (0..4).map(move |b| vec![a, b])).collect::<Vec<_>>();
    let mut sequences = Vec::new();
    for _ in 0..4 {
        for realized in 0..16 {
            let past = schedules[realized][0];
            let current = schedules[realized][1];
            let a = current & 1;
            let b = current >> 1;
            let mean = (2 * a + 3 * b + 4 * a * b) as f64
                + 1.5 * (past & 1) as f64
                + 2.0 * (past >> 1) as f64;
            sequences.push(antecedent_estimate::switchback::HistoryAssignment {
                schedules: schedules.clone(),
                probabilities: vec![1.0 / 16.0; 16],
                realized,
                outcomes: vec![0.0, mean],
            });
        }
    }
    let design = JointSwitchbackDesign {
        worlds: schedules.iter().enumerate().map(|(i, h)| (format!("h{i}"), h.clone())).collect(),
        carryover: 1,
        no_anticipation: "future assignments do not affect current outcomes".into(),
        finite_carryover: "response depends only on current and preceding joint arm".into(),
        sequence_isolation: "independently randomized noninterfering sequences".into(),
        population: "eligible terminal periods over these independent sequences".into(),
    };
    let prepared = PreparedNativeJointDesign::compile(
        JointDesignRecipe::Switchback { design: design.clone(), sequences: sequences.clone() },
        &context(),
    )
    .unwrap();
    assert_eq!(prepared.identification().target_kind, "eligible_period_joint_history_mean");
    let JointNativeDesignLaw::Moments(
        antecedent_estimate::joint_design::compiled::JointDesignLaw::Frequentist(fit),
    ) = prepared.point.as_ref()
    else {
        panic!("native design moment output")
    };
    assert_eq!(&fit.means[..4], &[0.0, 2.0, 3.0, 9.0]);
    assert!(
        (fit.contrast(&{
            let mut c = vec![0.0; 16];
            c[..4].copy_from_slice(&[1.0, -1.0, -1.0, 1.0]);
            c
        })
        .unwrap()
        .0 - 4.0)
            .abs()
            < 1e-12
    );
    replay(&prepared, JointDesignInference::Frequentist);
    let prior = SwitchbackGaussianPrior {
        mean: vec![0.0; 16],
        variance: vec![100.0; 16],
        sequence_variance: vec![1.0; sequences.len()],
        sequence_rho: vec![0.0; sequences.len()],
        draws: 8000,
        seed: 445,
    };
    let posterior = prepared
        .execute(JointDesignInference::SwitchbackGaussian { prior: prior.clone() }, &context())
        .unwrap();
    let JointNativeDesignLaw::Moments(
        antecedent_estimate::joint_design::compiled::JointDesignLaw::Bayesian(law),
    ) = posterior.law()
    else {
        panic!("actual posterior")
    };
    for (j, mean) in law.estimate.means.iter().enumerate() {
        let reference = fit.means[j] * 4.0 / 4.01;
        assert!((mean - reference).abs() < 6.0 * (1.0 / 4.01 / 8000.0f64).sqrt());
    }
    replay(&prepared, JointDesignInference::SwitchbackGaussian { prior });
    let mut missing = design;
    missing.finite_carryover.clear();
    assert!(
        PreparedNativeJointDesign::compile(
            JointDesignRecipe::Switchback { design: missing, sequences },
            &context()
        )
        .is_err()
    );
}
#[test]
fn native_structural_iv_compiler_keeps_actual_polynomial_features_and_joint_first_stage_posterior()
{
    let mut units = Vec::new();
    for repeat in 0..4 {
        for a in [-1.0, 1.0] {
            for b in [-1.0, 1.0] {
                for v1 in [-1.0, 1.0] {
                    for v2 in [-1.0, 1.0] {
                        for e in [-1.0, 1.0] {
                            let d1 = a + v1;
                            let d2 = b + v2;
                            units.push(JointInstrumentedUnit {
                                id: format!("{repeat}/{a}/{b}/{v1}/{v2}/{e}"),
                                endogenous: vec![d1, d2, d1 * d2],
                                instruments: vec![a, b, a * b],
                                exogenous: vec![1.0],
                                outcome: 2.0 * d1
                                    + 3.0 * d2
                                    + 4.0 * d1 * d2
                                    + 5.0
                                    + 0.6 * v1
                                    + 0.4 * v2
                                    + e,
                            });
                        }
                    }
                }
            }
        }
    }
    let design = JointInstrumentedDesign {
        contrasts: vec![
            ("a".into(), vec![1.0, 0.0, 0.0]),
            ("ab".into(), vec![0.0, 0.0, 1.0]),
            ("all".into(), vec![1.0, 1.0, 1.0]),
        ],
        structural_model: "homogeneous structural response with actual D1,D2,D1D2 features".into(),
        moment_restrictions:
            "Z=(A,B,AB) independent raw-dose/outcome disturbances; excluded structural instruments"
                .into(),
        population: "original independent IV population".into(),
    };
    let prepared = PreparedNativeJointDesign::compile(
        JointDesignRecipe::Instrumented { design: design.clone(), units: units.clone() },
        &context(),
    )
    .unwrap();
    assert_eq!(prepared.identification().target_kind, "homogeneous_structural_feature_contrast");
    replay(&prepared, JointDesignInference::Frequentist);
    let prior = JointIvGaussianPrior {
        raw_feature_indices: vec![0, 1],
        structural_features: vec![vec![1, 0], vec![0, 1], vec![1, 1]],
        parameter_mean: vec![0.0; 12],
        parameter_variance: vec![1000.0; 12],
        disturbance_covariance: vec![1.0, 0.0, 0.6, 0.0, 1.0, 0.4, 0.6, 0.4, 1.52],
        draws: 8000,
        seed: 446,
    };
    let posterior = prepared
        .execute(JointDesignInference::InstrumentedGaussian { prior: prior.clone() }, &context())
        .unwrap();
    let JointNativeDesignLaw::Moments(
        antecedent_estimate::joint_design::compiled::JointDesignLaw::Bayesian(law),
    ) = posterior.law()
    else {
        panic!("joint posterior")
    };
    for (got, want) in law.estimate.means.iter().zip([2.0, 4.0, 9.0]) {
        assert!((got - want).abs() < 0.08);
    }
    replay(&prepared, JointDesignInference::InstrumentedGaussian { prior: prior.clone() });
    let model = fit_instrumented_joint_polynomial(
        &design,
        &units,
        JointStructuralFeatureBinding {
            raw_feature_indices: prior.raw_feature_indices.clone(),
            powers: prior.structural_features.clone(),
        },
    )
    .unwrap();
    assert!((model.world_mean(&[1.0, 1.0]).unwrap().0 - 14.0).abs() < 1e-9);
    units[0].endogenous[2] += 1.0;
    assert!(instrumented_joint_coefficient_posterior(&design, &units, &prior).is_err());
}
#[test]
fn native_cohort_compiler_preserves_joint_regime_atts_and_complete_likelihood() {
    let mut units = Vec::new();
    for (regime, effect) in [("00", 0.0), ("10", 2.0), ("01", 3.0), ("11", 9.0)] {
        for i in 0..4 {
            units.push(JointRolloutUnit {
                id: format!("{regime}/{i}"),
                regime: regime.into(),
                adoption: (regime != "00").then_some(2),
                outcomes: vec![(1, i as f64), (2, i as f64 + 5.0 + effect)],
            });
        }
    }
    let design = JointRolloutDesign {
        cells: ["10", "01", "11"]
            .into_iter()
            .map(|r| JointRolloutCell {
                key: r.into(),
                regime: r.into(),
                adoption: 2,
                time: 2,
                baseline: 1,
            })
            .collect(),
        no_anticipation: "no effects before period2".into(),
        parallel_trends: "common untreated calendar increment".into(),
        consistency_and_isolation: "complete joint regimes, independent units".into(),
        population: "adopting regime cohorts".into(),
    };
    let prepared = PreparedNativeJointDesign::compile(
        JointDesignRecipe::DifferenceInDifferences { design: design.clone(), units: units.clone() },
        &context(),
    )
    .unwrap();
    replay(&prepared, JointDesignInference::Frequentist);
    let prior = JointRolloutGaussianPrior {
        unit_mean: 0.0,
        unit_variance: 1e6,
        time_mean: 0.0,
        time_variance: 1e6,
        effect_mean: vec![0.0; 3],
        effect_variance: vec![1e6; 3],
        outcome_variance: 1.0,
        ar1_rho: 0.0,
        draws: 10000,
        seed: 447,
    };
    let result = prepared
        .execute(JointDesignInference::RolloutGaussian { prior: prior.clone() }, &context())
        .unwrap();
    let JointNativeDesignLaw::Moments(
        antecedent_estimate::joint_design::compiled::JointDesignLaw::Bayesian(law),
    ) = result.law()
    else {
        panic!("joint posterior")
    };
    for (got, want) in law.estimate.means.iter().zip([2.0, 3.0, 9.0]) {
        assert!((got - want).abs() < 0.1);
    }
    assert!((law.estimate.covariance[1] - 0.5).abs() < 0.05);
    replay(&prepared, JointDesignInference::RolloutGaussian { prior });
    let mut bad = design;
    bad.parallel_trends.clear();
    assert!(
        PreparedNativeJointDesign::compile(
            JointDesignRecipe::DifferenceInDifferences { design: bad, units },
            &context()
        )
        .is_err()
    );
}

fn synthetic_fixture() -> (JointSyntheticDesign, Vec<JointSyntheticObservation>) {
    let mut data = Vec::new();
    for time in 1..=16 {
        let x = time as f64;
        let donors =
            [0.2 * x + (0.5 * x).sin(), 0.1 * x + (0.3 * x).cos(), 0.3 * x + (time % 5) as f64];
        for (j, y) in donors.iter().enumerate() {
            data.push(JointSyntheticObservation { unit: format!("d{j}"), time, outcome: *y });
        }
        for (index, effect) in [4.0, 8.0].iter().enumerate() {
            data.push(JointSyntheticObservation {
                unit: format!("treated{index}"),
                time,
                outcome: donors.iter().sum::<f64>() / 3.0 + if time >= 15 { *effect } else { 0.0 },
            });
        }
    }
    let design = JointSyntheticDesign {
        worlds: (0..2)
            .map(|index| JointSyntheticWorld {
                key: format!("joint{index}"),
                treated_unit: format!("treated{index}"),
                intervention: 15,
                donors: (0..3).map(|j| format!("d{j}")).collect(),
            })
            .collect(),
        counterfactual_model: "untreated response remains the preperiod simplex donor combination"
            .into(),
        donor_isolation: "all donors unaffected; no anticipation or interunit contamination".into(),
        population: "each original treated-unit rollout, postperiod15:16".into(),
    };
    (design, data)
}
#[test]
fn native_synthetic_compiler_preserves_donor_worlds_fit_and_joint_conformal_region() {
    let (design, data) = synthetic_fixture();
    let prepared = PreparedNativeJointDesign::compile(
        JointDesignRecipe::SyntheticControl { design: design.clone(), observations: data.clone() },
        &context(),
    )
    .unwrap();
    let JointNativeDesignLaw::SyntheticPoint(points) = prepared.point.as_ref() else {
        panic!("point-only target")
    };
    for (fit, want) in points.iter().zip([4.0, 8.0]) {
        assert!((fit.effect - want).abs() < 1e-7);
        assert!(fit.preperiod_rmse < 1e-7);
        assert_eq!(fit.donor_weights.len(), 3);
    }
    replay(&prepared, JointDesignInference::Frequentist);
    let inference = JointDesignInference::SyntheticConformal {
        joint_level: 0.8,
        residual_exchangeability:
            "exchangeable untreated counterfactual residuals across retained pre/post periods"
                .into(),
    };
    let result = prepared.execute(inference.clone(), &context()).unwrap();
    let JointNativeDesignLaw::SyntheticConformal(region) = result.law() else {
        panic!("joint conformal region")
    };
    let (effect, interval) = region.contrast_interval(&[-1.0, 1.0]).unwrap();
    assert!((effect - 4.0).abs() < 1e-7);
    assert!(interval[0] <= 4.0 + 1e-5 && interval[1] >= 4.0 - 1e-5);
    replay(&prepared, inference);
    let mut invalid = design;
    invalid.donor_isolation.clear();
    assert!(
        PreparedNativeJointDesign::compile(
            JointDesignRecipe::SyntheticControl { design: invalid, observations: data },
            &context()
        )
        .is_err()
    );
}
#[test]
fn native_synthetic_joint_posterior_uses_actual_convergence_gated_donor_likelihood() {
    let (design, data) = synthetic_fixture();
    let prepared = PreparedNativeJointDesign::compile(
        JointDesignRecipe::SyntheticControl { design, observations: data },
        &context(),
    )
    .unwrap();
    let prior = JointSyntheticPrior {
        dirichlet_concentration: 10.0,
        variance_shape: 4.0,
        variance_scale: 0.1,
        ar1_rho: 0.0,
        burn_in: 2000,
        draws: 8000,
        seed: 448,
        independent_treated_residuals:
            "independent treated counterfactual errors conditional on observed donors".into(),
    };
    let result = prepared
        .execute(JointDesignInference::SyntheticBayesian { prior: prior.clone() }, &context())
        .unwrap();
    let JointNativeDesignLaw::Moments(
        antecedent_estimate::joint_design::compiled::JointDesignLaw::Bayesian(law),
    ) = result.law()
    else {
        panic!("native joint posterior")
    };
    for (mean, want) in law.estimate.means.iter().zip([4.0, 8.0]) {
        assert!((mean - want).abs() < 0.15);
    }
    assert!(law.estimate.covariance[0] > 0.0);
    assert!(law.estimate.covariance[3] > 0.0);
    replay(&prepared, JointDesignInference::SyntheticBayesian { prior });
}

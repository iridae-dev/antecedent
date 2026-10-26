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
        &antecedent_core::ExecutionContext::for_tests(1),
    )
    .unwrap();
    assert!((model.world_mean(&[1.0, 1.0]).unwrap().0 - 14.0).abs() < 1e-9);
    units[0].endogenous[2] += 1.0;
    assert!(
        instrumented_joint_coefficient_posterior(
            &design,
            &units,
            &prior,
            &antecedent_core::ExecutionContext::for_tests(1)
        )
        .is_err()
    );
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

fn switchback_chain_fixture() -> (JointDesignRecipe, JointDesignInference, Vec<f64>) {
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
    let prior = SwitchbackGaussianPrior {
        mean: vec![0.0; 16],
        variance: vec![100.0; 16],
        sequence_variance: vec![1.0; sequences.len()],
        sequence_rho: vec![0.0; sequences.len()],
        draws: 8000,
        seed: 445,
    };
    (
        JointDesignRecipe::Switchback { design, sequences },
        JointDesignInference::SwitchbackGaussian { prior },
        {
            let mut c = vec![0.0; 16];
            c[..4].copy_from_slice(&[1.0, -1.0, -1.0, 1.0]);
            c
        },
    )
}
fn iv_chain_fixture() -> (JointDesignRecipe, JointDesignInference, Vec<f64>) {
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
    let prior = JointIvGaussianPrior {
        raw_feature_indices: vec![0, 1],
        structural_features: vec![vec![1, 0], vec![0, 1], vec![1, 1]],
        parameter_mean: vec![0.0; 12],
        parameter_variance: vec![1000.0; 12],
        disturbance_covariance: vec![1.0, 0.0, 0.6, 0.0, 1.0, 0.4, 0.6, 0.4, 1.52],
        draws: 8000,
        seed: 446,
    };
    (
        JointDesignRecipe::Instrumented { design, units },
        JointDesignInference::InstrumentedGaussian { prior },
        vec![0.0, 0.0, 1.0],
    )
}
fn did_chain_fixture() -> (JointDesignRecipe, JointDesignInference, Vec<f64>) {
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
    (
        JointDesignRecipe::DifferenceInDifferences { design, units },
        JointDesignInference::RolloutGaussian { prior },
        vec![-1.0, -1.0, 1.0],
    )
}
fn synthetic_chain_fixture() -> (JointDesignRecipe, JointDesignInference, Vec<f64>) {
    let (design, data) = synthetic_fixture();
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
    (
        JointDesignRecipe::SyntheticControl { design, observations: data },
        JointDesignInference::SyntheticBayesian { prior },
        vec![-1.0, 1.0],
    )
}
pub(crate) fn gaussian_longitudinal_chain_fixture()
-> (JointDesignRecipe, JointDesignInference, Vec<f64>) {
    use antecedent_estimate::joint_design::gaussian_longitudinal::*;
    let mut episodes = Vec::new();
    for repeat in 0..3 {
        for l in [-1.0, 1.0] {
            for v0 in [-1.0, 1.0] {
                for u in [-1.0, 1.0] {
                    for v1 in [-1.0, 1.0] {
                        for e in [-1.0, 1.0] {
                            let a0 = 0.6 * l + v0;
                            let l1 = 1.0 + 2.0 * a0 + 0.5 * l + u;
                            let a1 = 0.4 * l1 + 0.2 * a0 + v1;
                            episodes.push(GaussianLongitudinalEpisode {
                                id: format!("{repeat}/{l}/{v0}/{u}/{v1}/{e}"),
                                covariates: vec![vec![l], vec![l1]],
                                treatments: vec![vec![a0], vec![a1]],
                                outcome: 3.0
                                    + a0
                                    + 2.0 * a1
                                    + 4.0 * l1
                                    + 0.3 * l
                                    + 5.0 * a0 * a1
                                    + 0.2 * l1 * a1
                                    + e,
                            });
                        }
                    }
                }
            }
        }
    }
    let mut model = GaussianLongitudinalModel::full_history(2, 1, 1);
    model.outcome.push(GaussianHistoryFeature::TreatmentProduct {
        first_time: 0,
        first_channel: 0,
        second_time: 1,
        second_channel: 0,
    });
    model.outcome.push(GaussianHistoryFeature::CovariateTreatment {
        covariate_time: 1,
        coordinate: 0,
        treatment_time: 1,
        channel: 0,
    });
    let design = GaussianLongitudinalDesign {
        channel_variables: vec![1],
        covariate_variables: vec![2],
        worlds: [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
            .into_iter()
            .enumerate()
            .map(|(i, (a, b))| GaussianLongitudinalWorld {
                key: i.to_string(),
                treatments: vec![vec![a], vec![b]],
            })
            .collect(),
        assignment_support: vec![GaussianAssignmentSupport::Real],
        sequential_exchangeability: "full recorded histories suffice before each joint assignment"
            .into(),
        consistency: "stable continuous intervention versions".into(),
        population: "original independent fullhistory source episodes".into(),
        conditional_model:
            "conditional Gaussian full-history state/outcome and treatment-policy mechanisms".into(),
        model,
    };
    (
        JointDesignRecipe::GaussianLongitudinal { design, episodes },
        JointDesignInference::GaussianLongitudinalBayesian {
            prior: GaussianLongitudinalPrior {
                coefficient_scale: 1e6,
                variance_shape: 2.0,
                variance_scale: 1.0,
                draws: 512,
                seed: 556,
            },
        },
        vec![1.0, -1.0, -1.0, 1.0],
    )
}
fn weighted_geo_chain_fixture() -> (JointDesignRecipe, JointDesignInference, Vec<f64>) {
    use antecedent_estimate::joint_design::randomized::*;
    let design = JointRandomizedDesign {
        worlds: vec!["reference".into(), "joint".into()],
        sampling_unit: "geo".into(),
        population: "original unequal weighted geo census".into(),
        consistency: "wholeprogram assignment within each geo".into(),
        between_unit_isolation: "no spillovers between independently randomized geos".into(),
    };
    let units = (0..8)
        .map(|i| JointRandomizedUnit {
            id: format!("geo{i}"),
            arm: i % 2,
            probabilities: vec![0.5; 2],
            outcome: if i % 2 == 0 { 2.0 } else { 5.0 },
            population_weight: (i / 2 + 1) as f64,
        })
        .collect();
    let prior = RandomizedFinitePopulationPrior {
        arm: RandomizedGaussianPrior {
            mean: vec![2.0, 5.0],
            variance: vec![4.0; 2],
            outcome_variance: vec![1.0; 8],
            draws: 1024,
            seed: 557,
        },
        heterogeneity_variance: vec![2.0; 2],
    };
    (
        JointDesignRecipe::Randomized { design, units },
        JointDesignInference::RandomizedFinitePopulationGaussian { prior },
        vec![-1.0, 1.0],
    )
}
pub(crate) fn native_experiment_fixtures()
-> Vec<(JointDesignRecipe, JointDesignInference, Vec<f64>)> {
    vec![
        switchback_chain_fixture(),
        iv_chain_fixture(),
        did_chain_fixture(),
        synthetic_chain_fixture(),
        gaussian_longitudinal_chain_fixture(),
        weighted_geo_chain_fixture(),
    ]
}

#[test]
fn native_continuous_history_compiler_preserves_complete_refits_and_posterior_replay() {
    let ctx = ExecutionContext::for_tests(558);
    let (recipe, bayesian, contrast) = gaussian_longitudinal_chain_fixture();
    let prepared = PreparedNativeJointDesign::compile(recipe, &ctx).unwrap();
    let f = prepared
        .execute(
            JointDesignInference::GaussianLongitudinalBootstrap { replicates: 32, seed: 559 },
            &ctx,
        )
        .unwrap();
    let JointNativeDesignLaw::Moments(
        antecedent_estimate::joint_design::compiled::JointDesignLaw::FrequentistBootstrap {
            estimate,
            realizations,
            failures,
        },
    ) = f.law()
    else {
        panic!("full procedure slots required")
    };
    assert_eq!(realizations.len(), 32);
    assert_eq!(failures.len(), 32);
    assert!(realizations.iter().all(Option::is_some));
    for (actual, expected) in estimate.means.iter().zip([7., 16., 9.2, 23.6]) {
        assert!((actual - expected).abs() < 1e-8);
    }
    assert!((estimate.contrast(&contrast).unwrap().0 - 5.4).abs() < 1e-8);
    let b = prepared.execute(bayesian, &ctx).unwrap();
    let artifact = b.export_artifact().unwrap();
    assert_eq!(replay_native_joint_design(&artifact, &ctx).unwrap(), *b.law());
    prepared.consume_current(&artifact, &ctx).unwrap();
}
#[test]
fn native_local_iv_preserves_complier_target_and_refuses_missing_monotonicity() {
    use antecedent_estimate::joint_design::local_instrument::*;
    let ctx = ExecutionContext::for_tests(560);
    let design = LocalInstrumentDesign {
        key: "program encouragement compliers".into(),
        population: "randomized encouragement source".into(),
        exclusion: "encouragement affects outcome only through program receipt".into(),
        monotonicity: "no defiers".into(),
        consistency_and_isolation: "consistent complete program; independent units".into(),
    };
    let units = (0..800)
        .map(|i| {
            let assigned = i % 2 == 0;
            let received = assigned && (i / 2) % 2 == 0;
            LocalInstrumentUnit {
                id: format!("unit{i}"),
                assigned,
                received,
                probability: 0.5,
                outcome: if received { 3.0 } else { 0.0 },
            }
        })
        .collect();
    let recipe = JointDesignRecipe::InstrumentedLocal { design, units };
    let p = PreparedNativeJointDesign::compile(recipe.clone(), &ctx).unwrap();
    assert_eq!(p.identification().target_kind, "encouragement_specific_complier_average_effect");
    assert!(!p.identification().required_assumptions.iter().any(|s| s.contains("homogeneous")));
    let f = p.execute(JointDesignInference::Frequentist, &ctx).unwrap();
    let JointNativeDesignLaw::LocalComplier(estimate) = f.law() else {
        panic!("local target lost")
    };
    assert_eq!(estimate.effect, 3.0);
    assert_eq!(estimate.first_stage, 0.5);
    let b = p
        .execute(JointDesignInference::LocalComplierBayesian { draws: 512, seed: 560 }, &ctx)
        .unwrap();
    let JointNativeDesignLaw::LocalComplier(posterior) = b.law() else {
        panic!("local target lost")
    };
    for [ratio, numerator, denominator] in posterior.posterior.as_ref().unwrap() {
        assert!((ratio - 3.).abs() < 1e-10);
        assert!((numerator / denominator - ratio).abs() < 1e-12);
    }
    let artifact = b.export_artifact().unwrap();
    assert_eq!(replay_native_joint_design(&artifact, &ctx).unwrap(), *b.law());
    p.consume_current(&artifact, &ctx).unwrap();
    let JointDesignRecipe::InstrumentedLocal { mut design, units } = recipe else { unreachable!() };
    design.monotonicity.clear();
    assert!(
        PreparedNativeJointDesign::compile(
            JointDesignRecipe::InstrumentedLocal { design, units },
            &ctx
        )
        .is_err()
    );
}

/// Registered Q08/E14 procedure. This reports measurements; execution grants no release license.
#[test]
#[ignore = "400 independent original-data repetitions per design; run only on the stable candidate"]
fn compiled_native_design_full_procedure_calibration() {
    use antecedent_core::execution::CausalRng;
    use antecedent_estimate::joint_design::compiled::JointDesignLaw;
    let labels = [
        "switchback",
        "instrumented_dose",
        "difference_in_differences",
        "synthetic_control",
        "continuous_longitudinal",
        "weighted_geo",
        "factorial_multi_arm",
        "finite_longitudinal",
        "local_complier",
    ];
    let truths = [4.0, 9.0, 4.0, 4.0, 5.4, 3.0, 1.0, 0.375, 3.0];
    let mut fixtures = native_experiment_fixtures();
    use antecedent_estimate::joint_design::randomized::{
        JointRandomizedDesign, JointRandomizedUnit, RandomizedGaussianPrior,
    };
    fixtures.push((
        JointDesignRecipe::Randomized {
            design: JointRandomizedDesign {
                worlds: (0..4).map(|i| format!("arm{i}")).collect(),
                sampling_unit: "participant".into(),
                population: "factorial original population".into(),
                consistency: "complete factorial joint arms".into(),
                between_unit_isolation: "independent participants".into(),
            },
            units: (0..128)
                .map(|i| JointRandomizedUnit {
                    id: format!("participant{i}"),
                    arm: i % 4,
                    probabilities: vec![0.25; 4],
                    outcome: [0., 1., 2., 4.][i % 4],
                    population_weight: 1.,
                })
                .collect(),
        },
        JointDesignInference::RandomizedGaussian {
            prior: RandomizedGaussianPrior {
                mean: vec![0.; 4],
                variance: vec![100.; 4],
                outcome_variance: vec![1.; 128],
                draws: 1024,
                seed: 559,
            },
        },
        vec![1., -1., -1., 1.],
    ));
    use antecedent_estimate::joint_design::longitudinal::{
        LongitudinalDirichletPrior, LongitudinalEpisode, LongitudinalJointDesign,
    };
    fixtures.push((
        JointDesignRecipe::Longitudinal {
            design: LongitudinalJointDesign {
                worlds: vec![("00".into(), vec![0, 0]), ("11".into(), vec![1, 1])],
                sequential_exchangeability:
                    "recorded past states suffice before each randomized joint arm".into(),
                consistency: "complete versions".into(),
                sampling_population: "independent binary source episodes".into(),
            },
            episodes: (0..512)
                .map(|i| LongitudinalEpisode {
                    id: format!("binary{i}"),
                    covariates: vec![0, 0],
                    assignments: vec![0, 0],
                    outcome: 0.,
                })
                .collect(),
        },
        JointDesignInference::LongitudinalDirichlet {
            prior: LongitudinalDirichletPrior {
                covariate_states: vec![vec![0, 1]; 2],
                outcome_values: vec![0., 1.],
                concentration: 1.,
                draws: 1024,
                seed: 561,
            },
        },
        vec![-1., 1.],
    ));
    use antecedent_estimate::joint_design::local_instrument::{
        LocalInstrumentDesign, LocalInstrumentUnit,
    };
    fixtures.push((
        JointDesignRecipe::InstrumentedLocal {
            design: LocalInstrumentDesign {
                key: "local".into(),
                population: "encouragement source compliers".into(),
                exclusion: "no direct encouragement effect".into(),
                monotonicity: "one-sided receipt; no defiers".into(),
                consistency_and_isolation: "independent complete versions".into(),
            },
            units: (0..800)
                .map(|i| LocalInstrumentUnit {
                    id: format!("local{i}"),
                    assigned: false,
                    received: false,
                    probability: 0.5,
                    outcome: 0.,
                })
                .collect(),
        },
        JointDesignInference::LocalComplierBayesian { draws: 1024, seed: 562 },
        vec![1.],
    ));
    for (cell, (template, inference, contrast)) in fixtures.into_iter().enumerate() {
        let mut counts = [0usize; 6]; // F successes, F covers, F failures, B successes, B covers, B failures
        for repetition in 0..400u64 {
            let mut rng = CausalRng::from_seed(60000 + cell as u64 * 1000 + repetition);
            let normal = |r: &mut CausalRng| antecedent_kernels::standard_normal(r);
            let mut recipe = template.clone();
            let mut target_truth = truths[cell];
            match &mut recipe {
                JointDesignRecipe::Switchback { sequences, .. } => {
                    for sequence in sequences {
                        let realized = (rng.next_f64() * sequence.schedules.len() as f64) as usize;
                        sequence.realized = realized;
                        let h = &sequence.schedules[realized];
                        let past = h[0];
                        let arm = h[1];
                        sequence.outcomes = vec![
                            normal(&mut rng),
                            (2 * (arm & 1) + 3 * (arm >> 1) + 4 * (arm & 1) * (arm >> 1)) as f64
                                + 1.5 * (past & 1) as f64
                                + 2.0 * (past >> 1) as f64
                                + normal(&mut rng),
                        ];
                    }
                }
                JointDesignRecipe::Instrumented { units, .. } => {
                    for unit in units {
                        let v1 = normal(&mut rng);
                        let v2 = normal(&mut rng);
                        let d1 = unit.instruments[0] + v1;
                        let d2 = unit.instruments[1] + v2;
                        unit.endogenous = vec![d1, d2, d1 * d2];
                        unit.outcome = 2.0 * d1
                            + 3.0 * d2
                            + 4.0 * d1 * d2
                            + 5.0
                            + 0.6 * v1
                            + 0.4 * v2
                            + normal(&mut rng);
                    }
                }
                JointDesignRecipe::DifferenceInDifferences { units, .. } => {
                    for unit in units {
                        for (_, y) in &mut unit.outcomes {
                            *y += normal(&mut rng);
                        }
                    }
                }
                JointDesignRecipe::SyntheticControl { observations, .. } => {
                    for row in observations {
                        if row.unit.starts_with("treated") {
                            row.outcome += 0.1 * normal(&mut rng);
                        }
                    }
                }
                JointDesignRecipe::GaussianLongitudinal { episodes, .. } => {
                    for episode in episodes {
                        let l0 = normal(&mut rng);
                        let a0 = 0.6 * l0 + normal(&mut rng);
                        let l1 = 1.0 + 2.0 * a0 + 0.5 * l0 + normal(&mut rng);
                        let a1 = 0.4 * l1 + 0.2 * a0 + normal(&mut rng);
                        episode.covariates = vec![vec![l0], vec![l1]];
                        episode.treatments = vec![vec![a0], vec![a1]];
                        episode.outcome = 3.0
                            + a0
                            + 2.0 * a1
                            + 4.0 * l1
                            + 0.3 * l0
                            + 5.0 * a0 * a1
                            + 0.2 * l1 * a1
                            + normal(&mut rng);
                    }
                }
                JointDesignRecipe::Randomized { units, .. } => {
                    let mut weighted_potential_contrast = 0.0;
                    let mut total_weight = 0.0;
                    for unit in units {
                        if cell == 6 {
                            unit.arm = (rng.next_f64() * 4.0) as usize;
                            unit.outcome = [0., 1., 2., 4.][unit.arm] + normal(&mut rng);
                        } else {
                            // Generate the entire original census potential-response vector first.
                            // Its realized weighted mean, not its population expectation, is the target.
                            let potential = [
                                2.0 + (2.0_f64).sqrt() * normal(&mut rng),
                                5.0 + (2.0_f64).sqrt() * normal(&mut rng),
                            ];
                            weighted_potential_contrast +=
                                unit.population_weight * (potential[1] - potential[0]);
                            total_weight += unit.population_weight;
                            unit.arm = usize::from(rng.next_f64() >= 0.5);
                            unit.outcome = potential[unit.arm] + normal(&mut rng);
                        }
                    }
                    if cell == 5 {
                        target_truth = weighted_potential_contrast / total_weight;
                    }
                }
                JointDesignRecipe::Longitudinal { episodes, .. } => {
                    for episode in episodes {
                        let l0 = usize::from(rng.next_f64() < 0.5);
                        let a0 = usize::from(rng.next_f64() < 0.5);
                        let l1 = usize::from(rng.next_f64() < 0.25 + 0.5 * a0 as f64);
                        let a1 = usize::from(rng.next_f64() < 0.5);
                        let y = f64::from(
                            rng.next_f64()
                                < 0.125 + 0.25 * l0 as f64 + 0.25 * l1 as f64 + 0.25 * a1 as f64,
                        );
                        episode.covariates = vec![l0, l1];
                        episode.assignments = vec![a0, a1];
                        episode.outcome = y;
                    }
                }
                JointDesignRecipe::InstrumentedLocal { units, .. } => {
                    for unit in units {
                        unit.assigned = rng.next_f64() < 0.5;
                        unit.received = unit.assigned && rng.next_f64() < 0.5;
                        unit.outcome = 3.0 * f64::from(unit.received) + normal(&mut rng);
                    }
                }
            }
            let ctx = ExecutionContext::for_tests(70000 + repetition);
            let prepared = match PreparedNativeJointDesign::compile(recipe, &ctx) {
                Ok(p) => p,
                Err(_) => {
                    counts[2] += 1;
                    counts[5] += 1;
                    continue;
                }
            };
            let fi = if cell == 3 {
                JointDesignInference::SyntheticConformal{joint_level:0.95,residual_exchangeability:"independent Gaussian treated residuals exchangeable across all observed time periods".into()}
            } else if cell == 4 {
                JointDesignInference::GaussianLongitudinalBootstrap {
                    replicates: 128,
                    seed: repetition + 80000,
                }
            } else {
                JointDesignInference::Frequentist
            };
            for (mode, procedure) in [fi, inference.clone()].into_iter().enumerate() {
                match prepared.execute(procedure, &ctx) {
                    Ok(result) => {
                        let interval = match result.law() {
                            JointNativeDesignLaw::Moments(JointDesignLaw::Bayesian(p)) => {
                                let mut values = p
                                    .draws
                                    .iter()
                                    .map(|d| {
                                        d.iter().zip(&contrast).map(|(v, c)| v * c).sum::<f64>()
                                    })
                                    .collect::<Vec<_>>();
                                values.sort_by(f64::total_cmp);
                                Some([
                                    values[values.len() / 40],
                                    values[(values.len() * 39 / 40).min(values.len() - 1)],
                                ])
                            }
                            JointNativeDesignLaw::Moments(law) => {
                                law.estimate().contrast(&contrast).ok().map(|(mean, var)| {
                                    let radius = 1.959963984540054 * var.max(0.0).sqrt();
                                    [mean - radius, mean + radius]
                                })
                            }
                            JointNativeDesignLaw::SyntheticConformal(region) => region
                                .contrast_interval(&contrast)
                                .ok()
                                .map(|(_, interval)| interval),
                            _ => None,
                        };
                        if let Some([lo, hi]) = interval {
                            counts[mode * 3] += 1;
                            counts[mode * 3 + 1] +=
                                usize::from(lo <= target_truth && target_truth <= hi);
                        } else {
                            counts[mode * 3 + 2] += 1;
                        }
                        let bytes = result.export_artifact().unwrap();
                        assert_eq!(
                            *result.law(),
                            replay_native_joint_design(&bytes, &ctx).unwrap()
                        );
                        prepared.consume_current(&bytes, &ctx).unwrap();
                    }
                    Err(_) => counts[mode * 3 + 2] += 1,
                }
            }
        }
        println!(
            "{{\"cell\":\"{}\",\"repetitions\":400,\"frequentist_success\":{},\"frequentist_covered\":{},\"frequentist_failed\":{},\"bayesian_success\":{},\"bayesian_covered\":{},\"bayesian_failed\":{},\"release_license\":false}}",
            labels[cell], counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]
        );
        assert_eq!(counts[0] + counts[2], 400);
        assert_eq!(counts[3] + counts[5], 400);
    }
}

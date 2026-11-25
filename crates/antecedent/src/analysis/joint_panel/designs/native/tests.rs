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
        // Observed deviation 0.0016 at 8000 draws under a diffuse prior.
        assert!((got - want).abs() < 0.01);
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
                outcomes: vec![(1, f64::from(i)), (2, f64::from(i) + 5.0 + effect)],
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
        // Observed deviation 0.023 (diffuse-prior Monte Carlo).
        assert!((got - want).abs() < 0.06);
    }
    // Observed 0.49999 against the stated 0.5.
    assert!((law.estimate.covariance[1] - 0.5).abs() < 0.002);
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
    synthetic_panel(16, 15)
}
/// Three unaffected donors and two treated units over `periods`, treated from `intervention`.
fn synthetic_panel(
    periods: i64,
    intervention: i64,
) -> (JointSyntheticDesign, Vec<JointSyntheticObservation>) {
    let mut data = Vec::new();
    for time in 1..=periods {
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
                outcome: donors.iter().sum::<f64>() / 3.0
                    + if time >= intervention { *effect } else { 0.0 },
            });
        }
    }
    let design = JointSyntheticDesign {
        worlds: (0..2)
            .map(|index| JointSyntheticWorld {
                key: format!("joint{index}"),
                treated_unit: format!("treated{index}"),
                intervention,
                donors: (0..3).map(|j| format!("d{j}")).collect(),
            })
            .collect(),
        counterfactual_model: "untreated response remains the preperiod simplex donor combination"
            .into(),
        donor_isolation: "all donors unaffected; no anticipation or interunit contamination".into(),
        population: format!(
            "each original treated-unit rollout, postperiod{intervention}:{periods}"
        ),
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
        // Observed deviation 0.002.
        assert!((mean - want).abs() < 0.01);
    }
    assert!(law.estimate.covariance[0] > 0.0);
    assert!(law.estimate.covariance[3] > 0.0);
    replay(&prepared, JointDesignInference::SyntheticBayesian { prior });
}

fn switchback_chain_fixture() -> (JointDesignRecipe, JointDesignInference, Vec<f64>) {
    switchback_chain_fixture_with(4)
}
/// `repeats` independent sequences at each of the sixteen two-period schedules.
fn switchback_chain_fixture_with(
    repeats: usize,
) -> (JointDesignRecipe, JointDesignInference, Vec<f64>) {
    let schedules = (0..4).flat_map(|a| (0..4).map(move |b| vec![a, b])).collect::<Vec<_>>();
    let mut sequences = Vec::new();
    for _ in 0..repeats {
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
                outcomes: vec![(1, f64::from(i)), (2, f64::from(i) + 5.0 + effect)],
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
    synthetic_chain_fixture_with(synthetic_fixture(), 0.1)
}
fn synthetic_chain_fixture_with(
    (design, data): (JointSyntheticDesign, Vec<JointSyntheticObservation>),
    variance_scale: f64,
) -> (JointDesignRecipe, JointDesignInference, Vec<f64>) {
    let prior = JointSyntheticPrior {
        dirichlet_concentration: 10.0,
        variance_shape: 4.0,
        variance_scale,
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
fn native_continuous_history_default_bootstrap_is_fixed_and_replays() {
    use antecedent_estimate::joint_design::compiled::JointDesignLaw;

    let (recipe, _, _) = gaussian_longitudinal_chain_fixture();
    let prepare_ctx = ExecutionContext::for_tests(558);
    let other_ctx = ExecutionContext::for_tests(992);
    let prepared = PreparedNativeJointDesign::compile(recipe.clone(), &prepare_ctx).unwrap();
    let default = prepared.execute(JointDesignInference::Frequentist, &prepare_ctx).unwrap();
    let JointNativeDesignLaw::Moments(JointDesignLaw::FrequentistBootstrap {
        estimate,
        realizations,
        failures,
    }) = default.law()
    else {
        panic!("default must retain complete bootstrap slots");
    };
    assert_eq!(
        estimate.uncertainty,
        antecedent_estimate::joint_design::JointUncertainty::FullProcedureBootstrap
    );
    assert_eq!(realizations.len(), 64);
    assert_eq!(failures.len(), 64);
    assert!(
        realizations
            .iter()
            .zip(failures)
            .all(|(draw, failure)| draw.is_some() != failure.is_some())
    );
    let explicit = prepared
        .execute(
            JointDesignInference::GaussianLongitudinalBootstrap { replicates: 64, seed: 0 },
            &other_ctx,
        )
        .unwrap();
    assert_eq!(default.law(), explicit.law());
    // The compatibility recipe belongs to inference, not the context RNG.
    assert_eq!(
        recipe.execute_native(&JointDesignInference::Frequentist, &other_ctx).unwrap(),
        *default.law()
    );
    let artifact = default.export_artifact().unwrap();
    assert_eq!(replay_native_joint_design(&artifact, &other_ctx).unwrap(), *default.law());
    prepared.consume_current(&artifact, &other_ctx).unwrap();

    let changed_seed = prepared
        .execute(
            JointDesignInference::GaussianLongitudinalBootstrap { replicates: 64, seed: 1 },
            &other_ctx,
        )
        .unwrap();
    let JointNativeDesignLaw::Moments(JointDesignLaw::FrequentistBootstrap {
        realizations: changed,
        ..
    }) = changed_seed.law()
    else {
        panic!("explicit bootstrap slots required");
    };
    assert_ne!(realizations, changed, "explicit seed selects a different stream on this fixture");
    let short = prepared
        .execute(
            JointDesignInference::GaussianLongitudinalBootstrap { replicates: 8, seed: 0 },
            &other_ctx,
        )
        .unwrap();
    let JointNativeDesignLaw::Moments(JointDesignLaw::FrequentistBootstrap {
        realizations: prefix,
        ..
    }) = short.law()
    else {
        panic!("explicit bootstrap slots required");
    };
    assert_eq!(prefix.as_slice(), &realizations[..8]);
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
#[allow(clippy::float_cmp, reason = "the exact deterministic fixture pins both effects")]
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

#[test]
fn the_design_proof_carries_the_prepared_identification_and_source_into_an_attempt() {
    let (recipe, _, _) = switchback_chain_fixture();
    let ctx = context();
    let prepared = PreparedNativeJointDesign::compile(recipe.clone(), &ctx).unwrap();
    let proof = prepared.panel_design_proof();
    let super::super::super::JointPanelDesignProof::V1(
        super::super::super::JointPanelDesignTheorem::Native { identification, source_dependency },
    ) = &proof
    else {
        panic!("a native design converts to the native theorem, not the randomized one");
    };
    // The proof is the prepared design's own target and source identity, unchanged.
    assert_eq!(identification, prepared.identification());
    assert_eq!(*source_dependency, prepared.dependency());
    // It is stable across compilations of the same recipe, and stays a proof the
    // design-attempt contract can carry.
    let again = PreparedNativeJointDesign::compile(recipe, &ctx).unwrap();
    assert_eq!(again.panel_design_proof(), proof);
    let attempt = super::super::super::JointPanelDesignAttempt {
        id: "native.switchback".into(),
        outcome: super::super::super::JointPanelDesignOutcome::Identified,
        detail: String::new(),
        next_attempt: None,
        target_identity: prepared.dependency(),
        worlds: Vec::new(),
        proof: Some(proof.clone()),
    };
    assert_eq!(attempt.proof.as_ref(), Some(&proof));
}

#[test]
fn site_joint_panel_native_design_dependency_is_the_ordered_recipe() {
    use crate::identity_laws::{all_distinct, ordered_laws};
    use antecedent_estimate::joint_design::local_instrument::{
        LocalInstrumentDesign, LocalInstrumentUnit,
    };
    let design = || LocalInstrumentDesign {
        key: "site encouragement compliers".into(),
        population: "site population".into(),
        exclusion: "no direct effect".into(),
        monotonicity: "no defiers".into(),
        consistency_and_isolation: "independent units".into(),
    };
    let unit = |i: usize| LocalInstrumentUnit {
        id: format!("unit{i}"),
        assigned: i % 2 == 0,
        received: i % 4 == 0,
        probability: 0.5,
        outcome: i as f64,
    };
    let recipe = |units: &[usize]| JointDesignRecipe::InstrumentedLocal {
        design: design(),
        units: units.iter().map(|&i| unit(i)).collect(),
    };
    // Units are read positionally by the native estimators and their seeded resampling.
    ordered_laws(&[0, 1, 2, 3], |units| native_dependency(&recipe(units)).ok());
    assert_eq!(
        native_dependency(&recipe(&[0, 1])).unwrap(),
        native_dependency(&recipe(&[0, 1])).unwrap()
    );
    let mut dependencies: Vec<SemanticDigest> = native_experiment_fixtures()
        .iter()
        .map(|(recipe, _, _)| native_dependency(recipe).unwrap())
        .collect();
    dependencies.push(native_dependency(&recipe(&[0, 1])).unwrap());
    dependencies.push(native_dependency(&recipe(&[0, 1, 2])).unwrap());
    all_distinct(&dependencies);
}

// SPDX-License-Identifier: MIT OR Apache-2.0
use super::likelihood::*;
use super::*;

pub(super) fn fixture() -> (PanelData, PanelDraw, serde_json::Value) {
    let truth: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../conformance/response/joint_panel/expected.json"
    ))
    .unwrap();
    let rows: Vec<Vec<_>> =
        include_str!("../../../../conformance/response/joint_panel/randomized_interaction.csv")
            .lines()
            .skip(1)
            .map(|line| line.split(',').collect())
            .collect();
    let groups: Vec<_> = truth["variants"]["interaction"]["latent_truth"]
        .as_array()
        .unwrap()
        .iter()
        .map(|group| {
            let id = group["group"].as_u64().unwrap().to_string();
            let source: Vec<_> = rows
                .iter()
                .filter(|row| row[0] == id && row[1].parse::<i32>().unwrap() >= 0)
                .collect();
            let history =
                |name: &str| [group[name][0].as_f64().unwrap(), group[name][1].as_f64().unwrap()];
            PanelGroup {
                id,
                history: vec![history("initial_a"), history("initial_b")],
                channels: vec![
                    source.iter().map(|row| row[4].parse().unwrap()).collect(),
                    source.iter().map(|row| row[5].parse().unwrap()).collect(),
                ],
                outcome: source.iter().map(|row| row[6].parse().unwrap()).collect(),
                confounder: source.iter().map(|row| row[3].parse().unwrap()).collect(),
                season: source.iter().map(|row| row[2].parse().unwrap()).collect(),
            }
        })
        .collect();
    let parameters = PanelParameters {
        intercept: 2.0,
        channels: vec![
            PanelChannelParameters { id: "0".into(), coefficient: 3.0, lag: [0.6, 0.3, 0.1] },
            PanelChannelParameters { id: "1".into(), coefficient: 2.0, lag: [0.5, 0.35, 0.15] },
        ],
        interactions: vec![PanelInteractionParameter { id: "ab".into(), coefficient: 1.5 }],
        confounder: 0.7,
        season_effects: vec![0.0, 0.2, -0.1, 0.1],
        half_saturation: 2.0,
        rho: 0.45,
        innovation_sd: 0.2,
        group_sd: 0.35,
    };
    let group_intercepts = truth["variants"]["interaction"]["latent_truth"]
        .as_array()
        .unwrap()
        .iter()
        .map(|group| group["intercept"].as_f64().unwrap())
        .collect();
    (
        PanelData {
            channels: vec![PanelChannel { id: "0".into() }, PanelChannel { id: "1".into() }],
            interactions: vec![PanelInteractionSpec {
                id: "ab".into(),
                inputs: PanelFeatureInputs::LaggedSaturatedChannels,
                expression: antecedent_io_core::query_wire::ValueExprWire::from_domain(
                    &antecedent_core::ValueExpr::var(antecedent_core::VariableId::from_raw(0)).mul(
                        antecedent_core::ValueExpr::var(antecedent_core::VariableId::from_raw(1)),
                    ),
                )
                .unwrap(),
            }],
            groups,
            seasons: 4,
            lag_order: 2,
        },
        PanelDraw { parameters, group_intercepts },
        truth,
    )
}
pub(super) fn programs(data: &PanelData) -> Vec<PanelProgram> {
    [("reference", [0.0, 0.0]), ("a", [4.0, 0.0]), ("b", [0.0, 4.0]), ("joint", [4.0, 4.0])]
        .into_iter()
        .map(|(id, levels)| PanelProgram {
            id: id.into(),
            channels: vec![
                PanelChannelProgram { id: "0".into(), values: vec![levels[0]; 24] },
                PanelChannelProgram { id: "1".into(), values: vec![levels[1]; 24] },
            ],
            scenarios: data
                .groups
                .iter()
                .map(|group| PanelScenario {
                    group: group.id.clone(),
                    history: data
                        .channels
                        .iter()
                        .zip(&group.history)
                        .map(|(channel, values)| PanelChannelHistory {
                            id: channel.id.clone(),
                            values: *values,
                        })
                        .collect(),
                    confounder: group.confounder.clone(),
                    season: group.season.clone(),
                })
                .collect(),
        })
        .collect()
}

#[test]
fn gaussian_random_intercept_integration_matches_dense_covariance() {
    let residuals: [f64; 3] = [0.2, -0.1, 0.7];
    let rho: f64 = 0.4;
    let sigma: f64 = 0.3;
    let tau: f64 = 0.5;
    let covariance: Vec<_> = (0_usize..3)
        .flat_map(|i| {
            (0_usize..3).map(move |j| {
                tau * tau
                    + sigma * sigma / (1.0 - rho * rho)
                        * rho.powi(i32::try_from(i.abs_diff(j)).unwrap())
            })
        })
        .collect();
    let lower = antecedent_stats::cholesky_spd(&covariance, 3).unwrap();
    let inverse = antecedent_kernels::chol_inverse(&lower, 3);
    let quadratic: f64 = (0..3)
        .map(|i| residuals[i] * (0..3).map(|j| inverse[i * 3 + j] * residuals[j]).sum::<f64>())
        .sum();
    let expected = -0.5
        * (3.0 * std::f64::consts::TAU.ln()
            + antecedent_stats::chol_log_det(&lower, 3)
            + quadratic);
    let actual = ar1_random_intercept(&residuals, rho, sigma, tau).unwrap();
    assert!((actual.log_likelihood - expected).abs() < 1e-12);
    assert!(actual.intercept_variance > 0.0 && actual.intercept_variance < tau * tau);
}

#[test]
fn shared_draw_response_matches_independent_panel_truth_and_initial_histories() {
    let (data, draw, truth) = fixture();
    let output =
        evaluate_panel_programs(&data, &draw, &programs(&data), &ExecutionContext::for_tests(19))
            .unwrap();
    for response in &output {
        let expected =
            truth["variants"]["interaction"]["world_means"][&response.program].as_array().unwrap();
        for (actual, expected) in response.means.iter().zip(expected) {
            assert!((actual - expected.as_f64().unwrap()).abs() < 1e-12);
        }
    }
    for time in 2..24 {
        let delta = output[3].means[time] - output[1].means[time] - output[2].means[time]
            + output[0].means[time];
        assert!((delta - 2.0 / 3.0).abs() < 1e-12);
    }
    let mut changed = programs(&data);
    changed[0].scenarios[0].group = "unfitted".into();
    assert!(
        evaluate_panel_programs(&data, &draw, &changed, &ExecutionContext::for_tests(19)).is_err()
    );
}

#[test]
fn source_identity_covers_data_histories_group_order_and_static_specialization() {
    let (data, point, _) = fixture();
    let identity = data.identity().unwrap();
    let mut changed = data.clone();
    changed.groups[0].history[0][0] += 1.0;
    assert_ne!(changed.identity().unwrap(), identity);
    changed = data.clone();
    changed.groups.swap(0, 1);
    assert_ne!(changed.identity().unwrap(), identity);
    let x = encode(&data, &point.parameters).unwrap();
    let decoded = decode(&data, &x).unwrap();
    assert!((decoded.half_saturation - point.parameters.half_saturation).abs() < 1e-14);
    assert_eq!(decoded.season_effects.len(), data.seasons);
    changed = data.clone();
    changed.lag_order = 0;
    let mut p = point.parameters;
    p.channels[0].lag = [1.0, 0.0, 0.0];
    p.channels[1].lag = [1.0, 0.0, 0.0];
    p.rho = 0.0;
    assert_eq!(encode(&changed, &p).unwrap().len(), x.len() - 5);
}

#[test]
fn failed_optimizer_slots_are_retained_without_a_fabricated_covariance() {
    let (data, draw, _) = fixture();
    let options = PanelFitOptions {
        optimizer: PanelOptimizerOptions { max_iterations: 1, gradient_tolerance: 1e-20 },
        inference: PanelInference::Frequentist { draws: 7 },
        initial: Some(draw.parameters),
    };
    let fit = fit_panel(&data, &options, &ExecutionContext::for_tests(19)).unwrap();
    assert!(!fit.diagnostics.optimizer_converged);
    assert_eq!(fit.draws.len(), 7);
    assert!(fit.draws.iter().all(Option::is_none));
    assert_eq!(fit.diagnostics.failed_draws, 7);
    assert_eq!(fit.diagnostics.cluster_rank_bound, Some(3));
    assert_eq!(fit.diagnostics.claim, "model_conditional_mechanism_only");
    assert_eq!(fit.diagnostics.calibration, "unmeasured");
    assert!(fit.point.group_intercepts.iter().any(|value| value.abs() > 0.0));
    assert!(fit.retained_bytes() > 0);
}

#[test]
fn actual_posterior_draws_include_conditional_random_effect_uncertainty_and_report_mixing() {
    let (data, draw, _) = fixture();
    let options = PanelFitOptions {
        optimizer: PanelOptimizerOptions { max_iterations: 1, gradient_tolerance: 1e-4 },
        inference: PanelInference::Bayesian {
            prior: PanelPrior::default(),
            sampler: PanelSamplingOptions {
                chains: 2,
                warmup: 8,
                draws: 16,
                proposal_scale: 0.001,
                max_rhat: 1.01,
                min_ess: 100.0,
            },
        },
        initial: Some(draw.parameters),
    };
    let fit = fit_panel(&data, &options, &ExecutionContext::for_tests(29)).unwrap();
    assert_eq!(fit.draws.len(), 32);
    assert_eq!(fit.diagnostics.posterior_converged, Some(false));
    let one = fit.draws[0].as_ref().unwrap();
    let two = fit.draws[1].as_ref().unwrap();
    assert_ne!(one.group_intercepts, two.group_intercepts);
    assert_eq!(one.group_intercepts.len(), data.groups.len());
    assert!(fit.diagnostics.uncertainty.contains("conditional Gaussian"));
}

#[test]
fn resource_refusal_and_cancellation_are_distinct_from_statistical_failure() {
    let (data, _, _) = fixture();
    let options = PanelFitOptions {
        optimizer: PanelOptimizerOptions::default(),
        inference: PanelInference::Frequentist { draws: 7 },
        initial: None,
    };
    let mut limited = ExecutionContext::for_tests(2);
    limited.set_memory_limit(Some(8));
    assert!(matches!(fit_panel(&data, &options, &limited), Err(PanelError::Resource(_))));
    let cancelled = ExecutionContext::for_tests(2);
    cancelled.cancellation.cancel();
    assert!(matches!(fit_panel(&data, &options, &cancelled), Err(PanelError::Cancelled)));
}

#[test]
fn configured_three_channel_programmable_features_use_exact_axis_and_shared_draw() {
    let (mut data, mut draw, _) = fixture();
    data.channels.push(PanelChannel { id: "9".into() });
    for group in &mut data.groups {
        group.channels.push(vec![2.0; group.outcome.len()]);
        group.history.push([2.0; 2]);
    }
    draw.parameters.channels.push(PanelChannelParameters {
        id: "9".into(),
        coefficient: 0.0,
        lag: [0.6, 0.3, 0.1],
    });
    let variable = |id| antecedent_core::ValueExpr::var(antecedent_core::VariableId::from_raw(id));
    let expression = variable(0)
        .mul(variable(1))
        .mul(variable(9))
        .add(variable(9).min(antecedent_core::ValueExpr::constant(0.25)));
    data.interactions.push(PanelInteractionSpec {
        id: "higher_order".into(),
        inputs: PanelFeatureInputs::LaggedSaturatedChannels,
        expression: antecedent_io_core::query_wire::ValueExprWire::from_domain(&expression)
            .unwrap(),
    });
    draw.parameters
        .interactions
        .push(PanelInteractionParameter { id: "higher_order".into(), coefficient: 2.0 });
    data.validate().unwrap();
    let exposures = [1.2_f64, 3.1, 2.0];
    let saturated = exposures.map(|value| value / (value + 2.0));
    let expected_feature = expression
        .evaluate(None, &|variable| match variable.raw() {
            0 => Some(saturated[0]),
            1 => Some(saturated[1]),
            9 => Some(saturated[2]),
            _ => None,
        })
        .unwrap();
    let expected = draw.parameters.intercept
        + 3.0 * saturated[0]
        + 2.0 * saturated[1]
        + 1.5 * saturated[0] * saturated[1]
        + 2.0 * expected_feature
        + 0.7 * 0.4
        + 0.2;
    let compiled = CompiledPanel::new(&data).unwrap();
    assert!((response(&compiled, &draw.parameters, &exposures, 0.4, 1) - expected).abs() < 1e-12);
    let encoded = encode(&data, &draw.parameters).unwrap();
    assert_eq!(encoded.len(), 20); // 3D + I + S + 5, not a fixed two-channel design.
    assert_eq!(decode(&data, &encoded).unwrap().channels.len(), 3);
    let mut programs = programs(&data);
    for program in &mut programs {
        // Explicit ID binding permits a schedule order different from the source axis.
        program.channels.insert(0, PanelChannelProgram { id: "9".into(), values: vec![2.0; 24] });
    }
    let output =
        evaluate_panel_programs(&data, &draw, &programs, &ExecutionContext::for_tests(1)).unwrap();
    let t = 4;
    let delta = output[3].means[t] - output[1].means[t] - output[2].means[t] + output[0].means[t];
    assert!((delta - (1.5 + 2.0 * 0.5) * 4.0 / 9.0).abs() < 1e-12);
    let mut unknown = data.clone();
    unknown.interactions[0].expression =
        antecedent_io_core::query_wire::ValueExprWire::from_domain(&variable(99)).unwrap();
    assert!(unknown.validate().is_err());
    let mut duplicate = data.clone();
    duplicate.interactions.push(data.interactions[0].clone());
    assert!(duplicate.validate().is_err());
    let mut invalid = programs.clone();
    invalid[0].scenarios[0].history.pop();
    assert!(
        evaluate_panel_programs(&data, &draw, &invalid, &ExecutionContext::for_tests(1)).is_err()
    );
    let before = data.identity().unwrap();
    data.interactions[1].expression = antecedent_io_core::query_wire::ValueExprWire::from_domain(
        &variable(9).max(antecedent_core::ValueExpr::constant(0.25)),
    )
    .unwrap();
    assert_ne!(data.identity().unwrap(), before);
}

#[test]
fn batched_law_preserves_shared_worlds_failed_slots_and_releases_its_output_reservation() {
    let (data, point, _) = fixture();
    let ctx = ExecutionContext::for_tests(43);
    let available = ctx.admission().available().bytes();
    let mut overflow = point.clone();
    overflow.parameters.intercept = 1e308;
    let draws = vec![Some(point.clone()), None, Some(overflow)];
    let programs = programs(&data);
    let single = evaluate_panel_programs(&data, &point, &programs, &ctx).unwrap();
    let law = evaluate_panel_law(&data, &point, &draws, &programs, &ctx).unwrap();
    assert_eq!(law.point(), single);
    assert_eq!(law.draws()[0].as_deref(), Some(single.as_slice()));
    assert!(law.draws()[1].is_none() && law.draws()[2].is_none());
    assert_eq!(available - ctx.admission().available().bytes(), law.retained_bytes());
    let retained = law.retained_bytes();
    drop(law);
    assert_eq!(ctx.admission().available().bytes(), available);
    let mut limited = ExecutionContext::for_tests(43);
    limited.set_memory_limit(Some(retained - 1));
    assert!(matches!(
        evaluate_panel_law(&data, &point, &draws, &programs, &limited),
        Err(PanelError::Resource(_))
    ));
    let cancelled = ExecutionContext::for_tests(43);
    cancelled.cancellation.cancel();
    assert!(matches!(
        evaluate_panel_law(&data, &point, &draws, &programs, &cancelled),
        Err(PanelError::Cancelled)
    ));
}

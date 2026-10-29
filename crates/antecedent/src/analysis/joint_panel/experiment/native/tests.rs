mod acceptance_record {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/response/joint_panel/acceptance_record.rs"
    ));
}
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::*;
use antecedent_contract::bridge::*;
use antecedent_contract::evidence_ledger::{ObservationId, SourceDeclaration, SourceKind};
use antecedent_design::DesignError;
use antecedent_design::decision::{DecisionProblem, Utility};
use antecedent_design::decision_contract::*;

#[derive(Clone)]
struct LinearGain {
    coefficients: Vec<f64>,
    threshold: f64,
}
impl Utility<usize, PanelExperimentDecisionState> for LinearGain {
    fn evaluate_batch(
        &self,
        actions: &[usize],
        states: &[PanelExperimentDecisionState],
        out: &mut [f64],
    ) -> Result<(), DesignError> {
        for (i, a) in actions.iter().enumerate() {
            for (j, state) in states.iter().enumerate() {
                out[i * states.len() + j] = if *a == 0 {
                    0.0
                } else {
                    state.responses.iter().zip(&self.coefficients).map(|(v, c)| v * c).sum::<f64>()
                        - self.threshold
                };
            }
        }
        Ok(())
    }
}
fn bridge(
    quantity: &ScientificQuantity,
    mean: f64,
    variance: f64,
    covariance: f64,
    translation: usize,
) -> BridgeContract {
    let mut source = quantity.clone();
    source.functional_id = "declared_experimental_mean".into();
    let setting = ExposureSetting {
        population_id: quantity.population_id.clone(),
        time_start: "native_summary:0".into(),
        time_end: "native_summary:0".into(),
        dose: 1.0,
        dose_units: "native_projection".into(),
        duration: 1.0,
        duration_units: "native_summary_window".into(),
        exposure_regime: quantity.regime_id.clone(),
    };
    let mut result = BridgeContract {
        kind: BridgeKind::ParameterInterventionResponse,
        source: BridgeEndpoint { quantity: source, class: QuantityClass::ModelParameter },
        target: BridgeEndpoint {
            quantity: quantity.clone(),
            class: QuantityClass::InterventionResponse,
        },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope {
            population_id: quantity.population_id.clone(),
            regime_id: quantity.regime_id.clone(),
        },
        mapping: BridgeMapping::Affine { intercept: 0.0, slope: mean },
        assumptions: vec![BridgeAssumption {
            id: "explicit_native_translation".into(),
            kind: AssumptionKind::ParameterInterpretation,
            statement: format!(
                "The supplied paired affine experimental map answers this exact {} without converting it to a population effect",
                quantity.functional_id
            ),
            evidence: Some("separately_supplied_mapping_model".into()),
        }],
        transport: None,
        mapping_uncertainty: Some(MappingUncertainty {
            id: "paired_finite_map".into(),
            intercept_variance: 0.0,
            slope_variance: variance,
            intercept_slope_covariance: 0.0,
        }),
        source_dependence: Some(SourceMappingDependence::SlopeCovariance {
            covariance,
            evidence: "retained_joint_parameter_and_mapping_slots".into(),
        }),
        estimand_bridge: Some(EstimandBridge {
            id: "native_estimand_translation".into(),
            experiment_id: "independent_experiment".into(),
            experiment: setting.clone(),
            target: setting,
            recency_delta: 0.0,
            recency_units: "native_summary_window".into(),
            spend_delta: 0.0,
            spend_units: "native_projection".into(),
        }),
    };
    match translation {
        0 => {}
        1 => {
            result.source.quantity.population_id = "external_experimental_population".into();
            result.evidence_scope.population_id = result.source.quantity.population_id.clone();
            result.estimand_bridge.as_mut().unwrap().experiment.population_id =
                result.source.quantity.population_id.clone();
        }
        2 => {
            result.source.quantity.horizon = 2;
            let e = result.estimand_bridge.as_mut().unwrap();
            e.experiment.time_start = "historical_period:0".into();
            e.experiment.time_end = "historical_period:1".into();
            e.experiment.duration = 2.0;
            e.recency_delta = 10.0;
        }
        3 => {
            let e = result.estimand_bridge.as_mut().unwrap();
            e.experiment.dose = 0.5;
            e.spend_delta = 0.5;
        }
        4 => {
            result.source.quantity.horizon = 2;
            let e = result.estimand_bridge.as_mut().unwrap();
            e.experiment.duration = 2.0;
        }
        _ => unreachable!(),
    }
    if [1, 2, 4].contains(&translation) {
        result.transport=Some(TransportabilityPremise {id:"explicit_native_transport".into(),statement:"This stated target-specific translation applies across the declared experimental and target population/time/duration settings".into(),established_in:result.evidence_scope.clone(),applies_to:BridgeScope {population_id:quantity.population_id.clone(),regime_id:quantity.regime_id.clone()},evidence:Some("declared_transport_evidence".into())});
    }
    result.check_licensing().unwrap();
    result
}
fn zero() -> UtilityExpr {
    UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Input(0))
}
#[test]
fn all_native_target_kinds_execute_full_experiment_matrix_against_independent_references() {
    let ctx = ExecutionContext::for_tests(445);
    let mut fixtures = super::super::super::designs::native_experiment_fixtures();
    fixtures.push(local_chain_fixture());
    assert_eq!(fixtures.len(), 7);
    let mut rows = 0;
    for (index, ((recipe, inference, coefficients), known_truth)) in
        fixtures.into_iter().zip([4.0, 9.0, 4.0, 4.0, 5.4, 3.0, 3.0]).enumerate()
    {
        let frequentist_inference = if index == 4 {
            antecedent_estimate::joint_design::compiled::JointDesignInference::GaussianLongitudinalBootstrap {replicates: 20, seed: 559}
        } else {
            antecedent_estimate::joint_design::compiled::JointDesignInference::Frequentist
        };
        let prepared =
            super::super::super::designs::PreparedNativeJointDesign::compile(recipe, &ctx).unwrap();
        let frequentist = prepared.execute(frequentist_inference, &ctx).unwrap();
        let result = prepared.execute(inference, &ctx).unwrap();
        let binding = NativeExperimentBinding {
            coefficients,
            outcome_units: "original_response_unit".into(),
            local_ratio_regularization: if index == 6 {
                Some(LocalRatioGaussianRegularization {max_abs_intercept: 1.0, min_abs_slope: 0.5, max_abs_slope: 2.0, assumption: "The complete paired mapping law is intercept zero and slope1+0.05tanh(R-center), globally within[0.95,1.05] regardless of the ratio tail".into()})
            } else {
                None
            },
        };
        let family = [
            "switchback",
            "instrumented_dose",
            "difference_in_differences",
            "synthetic_control",
            "sequential_observational",
            "geo_experiment",
            "instrumented_dose",
        ][index];
        if index == 6 {
            assert!(
                matches!(native_prior(&result, &binding), Err(JointPanelError::Unsupported(m))
                    if m.contains("cannot become a Bayesian experimental prior")),
                "unconditioned local ratio moments are not asserted"
            );
        }
        let original_law = reference_prior(&result, &binding);
        let projected = original_law
            .draws()
            .chunks_exact(binding.coefficients.len())
            .map(|r| r.iter().zip(&binding.coefficients).map(|(v, c)| v * c).sum::<f64>())
            .sum::<f64>()
            / original_law.n_draws() as f64;
        // Observed deviations of the projected prior mean from the separately derived truth are
        // [-0.0038, -0.0012, 0.0228, -0.0020, -0.0032, 0.0075, -0.0048]; each tolerance is about
        // four times its fixture's deviation, far under the 0.3 that admitted a wrong functional.
        let tolerance = [0.015, 0.006, 0.09, 0.008, 0.013, 0.03, 0.02][index];
        assert!(
            (projected - known_truth).abs() < tolerance,
            "native target functional must recover the separately derived fixture truth {known_truth}: {projected}"
        );
        let case = [
            "switchback",
            "instrumented_dose",
            "difference_in_differences",
            "synthetic_control",
            "native_gaussian_longitudinal",
            "native_weighted_geo",
            "native_local_complier",
        ][index];
        for uncertain in [false, true] {
            for translation in 0..5 {
                sampling_law_refusal(&frequentist, &binding, &ctx);
                acceptance_record::record(
                    &format!("{family}::frequentist"),
                    uncertain,
                    translation,
                    case,
                    &frequentist.dependency_identity().to_hex(),
                    serde_json::json!({"scientific_target":frequentist.prepared().identification(),"sampling_law_prior_guard":"expected_scoped_refusal","downstream_stages":"not_applicable_sampling_law_is_not_a_prior"}),
                );
                exercise(&result, &binding, uncertain, translation, family, case, &ctx);
                rows += 3;
            }
        }
    }
    assert_eq!(
        rows, 210,
        "all seven native scientific procedures × two mappings × five translations × three decision cases"
    );
}
fn exercise(
    original: &NativeJointDesignResult,
    binding: &NativeExperimentBinding,
    uncertain: bool,
    translation: usize,
    family: &str,
    case: &str,
    ctx: &ExecutionContext,
) {
    let prior = reference_prior(original, binding);
    let width = binding.coefficients.len();
    let values = prior
        .draws()
        .chunks_exact(width)
        .map(|row| row.iter().zip(&binding.coefficients).map(|(v, c)| v * c).sum::<f64>())
        .collect::<Vec<_>>();
    let n = values.len();
    let center = values.iter().sum::<f64>() / n as f64;
    let slopes = values
        .iter()
        .map(|v| if uncertain { 1.0 + 0.05 * (v - center).tanh() } else { 1.0 })
        .collect::<Vec<_>>();
    let mean = slopes.iter().sum::<f64>() / n as f64;
    let variance = slopes.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n as f64;
    let covariance =
        slopes.iter().zip(&values).map(|(s, v)| (s - mean) * (v - center)).sum::<f64>() / n as f64;
    let quantity = original.experiment_quantity(binding).unwrap();
    assert!(quantity.functional_id.starts_with(&original.prepared().identification().target_kind));
    let mut ledger = EvidenceLedger::new();
    let model = ledger
        .add_source(
            SourceDeclaration::new(
                original.dependency_identity().to_hex(),
                SourceKind::FittedModel,
            )
            .with_observations([ObservationId::from_key("original-native-data", b"sample")]),
        )
        .unwrap();
    let experiment = ledger
        .add_source(
            SourceDeclaration::new("independent_experiment", SourceKind::Experiment)
                .with_observations([ObservationId::from_key(
                    "new-independent-experiment",
                    b"sample",
                )]),
        )
        .unwrap();
    let mapping = ledger
        .add_source(
            SourceDeclaration::new("native_target_mapping", SourceKind::FittedModel)
                .with_observations([ObservationId::from_key("external-mapping", b"sample")]),
        )
        .unwrap();
    let observed = center + 0.2;
    let request = JointLawExperimentRequest {
        bridge: bridge(&quantity, mean, variance, covariance, translation),
        observed_mean: observed,
        observation_variance: 0.3,
        sample_size: 8,
        mapping: PanelExperimentMapping {
            dependency: original.dependency_identity(),
            intercepts: vec![0.0; n],
            slopes: slopes.clone(),
        },
        evidence: PanelExperimentEvidence { model, experiment, mapping, prior_origins: vec![] },
    };
    assert_ne!(
        original.dependency_identity(),
        original.prepared().dependency(),
        "exact posterior slots must bind prior/seed/law in addition to original data"
    );
    let mut stale = request.clone();
    stale.mapping.dependency = original.prepared().dependency();
    assert!(
        matches!(original.condition_native_experiment(binding.clone(), stale, &ledger, ctx),
            Err(JointPanelError::Unsupported(m)) if m.contains("another fitted source or numerical dependency")),
        "a source-data identity alone cannot authorize posterior mapping slots"
    );
    let mut unsupported = request.clone();
    unsupported.bridge.target.quantity.functional_id = "population_ate".into();
    assert!(
        matches!(original.condition_native_experiment(binding.clone(), unsupported, &ledger, ctx),
            Err(JointPanelError::Compile(m)) if m.contains("exact method-native functional, population and units")),
        "a native structural/cohort/donor target cannot silently acquire another population-effect meaning"
    );
    if binding.local_ratio_regularization.is_some() {
        let mut missing = binding.clone();
        missing.local_ratio_regularization = None;
        let mut bad_request = request.clone();
        bad_request.bridge.target.quantity = original.experiment_quantity(&missing).unwrap();
        assert!(
            matches!(original.condition_native_experiment(missing, bad_request, &ledger, ctx),
                Err(JointPanelError::Compile(m)) if m.contains("bridge_binding_mismatch")),
            "undefined ratio moments need a complete posterior regularization premise"
        );
        let mut invalid = binding.clone();
        invalid.local_ratio_regularization.as_mut().unwrap().max_abs_slope = 0.75;
        let mut bad_request = request.clone();
        bad_request.bridge.target.quantity = original.experiment_quantity(&invalid).unwrap();
        assert!(
            matches!(original.condition_native_experiment(invalid, bad_request, &ledger, ctx),
                Err(JointPanelError::Compile(m)) if m.contains("bridge_binding_mismatch")),
            "realizations outside claimed global support must refuse"
        );
    }
    let updated = original
        .condition_native_experiment(binding.clone(), request.clone(), &ledger, ctx)
        .unwrap();
    assert_eq!(updated.identification(), original.prepared().identification());
    let law = updated.numerical_law_source(&ledger).unwrap();
    let weights = law.metadata().weights.as_ref().unwrap();
    let mapped = values.iter().zip(&slopes).map(|(v, s)| v / s).collect::<Vec<_>>();
    let anchor = *mapped
        .iter()
        .min_by(|a, b| (observed - **a).abs().total_cmp(&(observed - **b).abs()))
        .unwrap();
    let likelihood = mapped
        .iter()
        .map(|m| ((m - anchor) * (2.0 * observed - m - anchor) / (2.0 * 0.3 / 8.0)).exp())
        .collect::<Vec<_>>();
    let mass = likelihood.iter().sum::<f64>();
    for (actual, reference) in weights.iter().zip(&likelihood) {
        assert!((actual - reference / mass).abs() < 1e-10);
    }
    for j in 0..width {
        let reference = prior
            .draws()
            .chunks_exact(width)
            .zip(&likelihood)
            .map(|(row, l)| row[j] * l / mass)
            .sum::<f64>();
        assert!((law.mean(j).unwrap() - reference).abs() < 1e-8);
    }
    let current = values.iter().zip(weights).map(|(v, w)| v * w).sum::<f64>();
    let coords = original.experiment_quantities(binding).unwrap();
    let expr =
        binding.coefficients.iter().enumerate().fold(UtilityExpr::Const(-current), |e, (i, c)| {
            UtilityExpr::sum(e, UtilityExpr::product(UtilityExpr::Const(*c), UtilityExpr::Input(i)))
        });
    let contract = DecisionContract {
        actions: vec![
            DecisionAction {
                id: "hold".into(),
                kind: ActionKind::Policy,
                inputs: coords.clone(),
                utility: zero(),
            },
            DecisionAction {
                id: "act".into(),
                kind: ActionKind::Policy,
                inputs: coords,
                utility: expr,
            },
        ],
        utility_units: quantity.units.clone(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: quantity.population_id.clone(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    };
    let tie = updated.decide(&contract, &ledger).unwrap();
    assert!(tie.result.actions.iter().all(|a| a.expected_utility.abs() < 1e-8));
    assert_eq!(tie.identification.target_kind, original.prepared().identification().target_kind);
    let perfect = values.iter().zip(weights).map(|(v, w)| w * (v - current).max(0.0)).sum::<f64>();
    assert!((tie.result.evpi.unwrap() - perfect).abs() < 1e-8);
    let mut constrained = contract.clone();
    constrained.constraints.push(HardConstraint {
        id: "binding_operational_capacity".into(),
        expr: UtilityExpr::sum(zero(), UtilityExpr::Const(1.0)),
        bound: 0.0,
        min_probability: 1.0,
        units: "operational_capacity".into(),
        applies_to: vec!["act".into()],
    });
    let restricted = updated.decide(&constrained, &ledger).unwrap();
    assert!(!restricted.result.actions[1].admissible);
    for gain in [-1e-6, 1e-6] {
        let mut weak = contract.clone();
        weak.actions[1].utility =
            UtilityExpr::sum(weak.actions[1].utility.clone(), UtilityExpr::Const(gain));
        let d = updated.decide(&weak, &ledger).unwrap();
        assert!((d.result.actions[1].expected_utility - gain).abs() < 1e-8);
    }
    let problem = DecisionProblem::new(
        vec![0, 1],
        Arc::new(LinearGain { coefficients: binding.coefficients.clone(), threshold: current }),
        vec![],
    );
    let candidates = [
        PanelFutureSample {
            id: "low_cost".into(),
            sample_size: 4,
            observation_variance: 1.0,
            cost_utility: Some(0.01),
        },
        PanelFutureSample {
            id: "high_cost".into(),
            sample_size: 4,
            observation_variance: 1.0,
            cost_utility: Some(0.1),
        },
    ];
    let ranking = updated.rank_next_samples(&problem, &candidates, 64, &ledger, ctx).unwrap();
    assert_eq!(ranking[0].id, "low_cost");
    assert_eq!(ranking[0].gross_evsi, ranking[1].gross_evsi);
    let feasible_actions = restricted
        .result
        .actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| a.admissible.then_some(i))
        .collect::<Vec<_>>();
    assert_eq!(feasible_actions, vec![0]);
    let constrained_problem = DecisionProblem::new(
        feasible_actions,
        Arc::new(LinearGain { coefficients: binding.coefficients.clone(), threshold: current }),
        vec![],
    );
    let constrained_ranking =
        updated.rank_next_samples(&constrained_problem, &candidates, 8, &ledger, ctx).unwrap();
    assert_ne!(
        constrained_ranking[0].admissible_action_identity,
        ranking[0].admissible_action_identity
    );
    assert!(
        ranking[0].gross_evsi > constrained_ranking[0].gross_evsi + 1e-12,
        "changing operational feasibility must change future information value"
    );
    for row in &constrained_ranking {
        assert!(row.gross_evsi.abs() < 1e-12);
        let cost = if row.id == "low_cost" { 0.01 } else { 0.1 };
        assert!((row.net_value.unwrap() + cost).abs() < 1e-12);
    }
    let (reference, error) = reference_evsi(&values, weights, &mapped, current);
    for row in &ranking {
        assert!(
            (row.gross_evsi - reference).abs() <= error + 6.0 * row.standard_error + 1e-9,
            "{}: {} vs reference{reference} +/-{}",
            original.prepared().identification().method,
            row.gross_evsi,
            error + 6.0 * row.standard_error
        );
    }
    let artifact = updated.export(&ledger, ctx).unwrap();
    fresh_replay(&artifact, law);
    let replay = NativeJointExperimentResult::replay(&artifact, ctx).unwrap();
    assert!(!replay.grants_current_authority());
    assert_eq!(replay.numerical_law().metadata(), law.metadata());
    assert_eq!(replay.numerical_law().draws(), law.draws());
    updated.consume_current(&artifact, &ledger, ctx).unwrap();
    // Another outcome does not inherit these weights or artifact authority.
    let mut different = request;
    different.observed_mean -= 0.4;
    let changed =
        original.condition_native_experiment(binding.clone(), different, &ledger, ctx).unwrap();
    assert!(matches!(changed.consume_current(&artifact, &ledger, ctx),
        Err(JointPanelError::Compile(m)) if m.contains("replay differs from caller-current")));
    let other = changed.numerical_law_source(&ledger).unwrap();
    assert!(
        other
            .metadata()
            .weights
            .as_ref()
            .unwrap()
            .iter()
            .zip(weights)
            .any(|(a, b)| (a - b).abs() > 1e-12)
    );
    ledger.withdraw_source(experiment, "withdrawn external evidence").unwrap();
    assert!(matches!(updated.consume_current(&artifact, &ledger, ctx),
        Err(JointPanelError::Unsupported(m)) if m.contains("missing, withdrawn or superseded")));
    let max_weight_error =
        weights.iter().zip(&likelihood).map(|(w, l)| (w - l / mass).abs()).fold(0.0_f64, f64::max);
    acceptance_record::record(
        &format!("{family}::bayesian"),
        uncertain,
        translation,
        case,
        &original.dependency_identity().to_hex(),
        serde_json::json!({"scientific_target":original.prepared().identification(),"local_ratio_regularization":binding.local_ratio_regularization,"constrained_next_study":{"evsi_reference":0.0,"evsi_absolute_error":constrained_ranking[0].gross_evsi.abs(),"quadrature_error_bound":0.0,"evsi_mcse":constrained_ranking[0].standard_error,"admissible_action_identity":constrained_ranking[0].admissible_action_identity.to_hex(),"operational_action_ids":[0],"net_value":constrained_ranking[0].net_value},"admissible_action_identity":ranking[0].admissible_action_identity.to_hex(),"original_target_truth":"checked","posterior_weight_max_error":max_weight_error,"posterior_coordinate_mean_tolerance":1e-8,"utility_and_evpi_tolerance":1e-8,"operational_exclusion":"checked","ties_and_near_ties":"checked","evsi_reference":reference,"quadrature_error_bound":error,"evsi_absolute_error":(ranking[0].gross_evsi-reference).abs(),"evsi_mcse":ranking[0].standard_error,"historical_replay":"fresh_process_checked","current_replay_and_withdrawal":"checked","changed_evidence":"changes_weights"}),
    );
}
fn reference_evsi(values: &[f64], weights: &[f64], means: &[f64], threshold: f64) -> (f64, f64) {
    let sigma = 0.5;
    let lower = means.iter().copied().fold(f64::INFINITY, f64::min) - 8.0 * sigma;
    let upper = means.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 8.0 * sigma;
    let step = (upper - lower) / 1024.0;
    let gains = values.iter().zip(weights).map(|(v, w)| w * (v - threshold)).collect::<Vec<_>>();
    let scale = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
    let mut future = 0.0;
    for cell in 0..1024 {
        let y = lower + (f64::from(cell) + 0.5) * step;
        future += step
            * means
                .iter()
                .zip(&gains)
                .map(|(m, g)| g * scale * (-0.5 * ((y - m) / sigma).powi(2)).exp())
                .sum::<f64>()
                .max(0.0);
    }
    let current = gains.iter().sum::<f64>().max(0.0);
    let absolute = gains.iter().map(|g| g.abs()).sum::<f64>();
    let error = step * 0.5 * absolute * (2.0 / std::f64::consts::PI).sqrt() / sigma
        + absolute * 2.0 * (-32.0_f64).exp() / (8.0 * (2.0 * std::f64::consts::PI).sqrt());
    (future - current, error)
}

fn sampling_law_refusal(
    original: &NativeJointDesignResult,
    binding: &NativeExperimentBinding,
    ctx: &ExecutionContext,
) {
    let quantity = original.experiment_quantity(binding).unwrap();
    let mut ledger = EvidenceLedger::new();
    let model = ledger
        .add_source(
            SourceDeclaration::new(
                original.dependency_identity().to_hex(),
                SourceKind::FittedModel,
            )
            .with_observations([ObservationId::from_key("f-model", b"sample")]),
        )
        .unwrap();
    let experiment = ledger
        .add_source(
            SourceDeclaration::new("new", SourceKind::Experiment)
                .with_observations([ObservationId::from_key("f-new", b"sample")]),
        )
        .unwrap();
    let mapping = ledger
        .add_source(
            SourceDeclaration::new("mapping", SourceKind::FittedModel)
                .with_observations([ObservationId::from_key("f-map", b"sample")]),
        )
        .unwrap();
    let request = JointLawExperimentRequest {
        bridge: bridge(&quantity, 1.0, 0.0, 0.0, 0),
        observed_mean: 0.0,
        observation_variance: 1.0,
        sample_size: 1,
        mapping: PanelExperimentMapping {
            dependency: original.dependency_identity(),
            intercepts: vec![0.0],
            slopes: vec![1.0],
        },
        evidence: PanelExperimentEvidence { model, experiment, mapping, prior_origins: vec![] },
    };
    assert!(matches!(
        original.condition_native_experiment(binding.clone(), request, &ledger, ctx),
        Err(JointPanelError::Unsupported(_))
    ));
}

fn fresh_replay(bytes: &[u8], law: &DistributionArtifact) {
    static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let directory = std::env::temp_dir()
        .join(format!("antecedent-native-target-experiment-{}-{sequence}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("execution.bin"), bytes).unwrap();
    let expected = (
        law.draws().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        law.metadata().weights.as_ref().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
    );
    std::fs::write(directory.join("expected.json"), serde_json::to_vec(&expected).unwrap())
        .unwrap();
    // The child must select this very test; its libtest name carries the module path.
    let prefix =
        module_path!().split_once("::").map_or(String::new(), |(_, rest)| format!("{rest}::"));
    let test_name = format!("{prefix}fresh_native_experiment_consumer");
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &test_name, "--nocapture"])
        .env("ANTECEDENT_NATIVE_TARGET_EXPERIMENT_REPLAY", &directory)
        .output()
        .unwrap();
    assert!(
        child.status.success(),
        "fresh native target replay: {}",
        String::from_utf8_lossy(&child.stderr)
    );
    assert!(
        String::from_utf8_lossy(&child.stdout).contains("running 1 test"),
        "the child selected no test: {}",
        String::from_utf8_lossy(&child.stdout)
    );
    assert!(directory.join("consumer-ran").exists());
    std::fs::remove_dir_all(directory).unwrap();
}
#[test]
fn fresh_native_experiment_consumer() {
    let Some(directory) = std::env::var_os("ANTECEDENT_NATIVE_TARGET_EXPERIMENT_REPLAY") else {
        return;
    };
    let directory = std::path::PathBuf::from(directory);
    let artifact = std::fs::read(directory.join("execution.bin")).unwrap();
    let expected: (Vec<u64>, Vec<u64>) =
        serde_json::from_slice(&std::fs::read(directory.join("expected.json")).unwrap()).unwrap();
    let replay =
        NativeJointExperimentResult::replay(&artifact, &ExecutionContext::for_tests(77)).unwrap();
    assert!(!replay.grants_current_authority());
    let law = replay.numerical_law();
    assert_eq!(law.draws().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.0);
    assert_eq!(
        law.metadata().weights.as_ref().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.1
    );
    std::fs::write(directory.join("consumer-ran"), b"verified").unwrap();
}

fn reference_prior(
    original: &NativeJointDesignResult,
    binding: &NativeExperimentBinding,
) -> DistributionArtifact {
    if let JointNativeDesignLaw::LocalComplier(local) = original.law() {
        let rows = local
            .posterior
            .as_ref()
            .expect("actual original local posterior")
            .iter()
            .map(|row| vec![row[0]])
            .collect::<Vec<_>>();
        native_draw_artifact(original, binding, &rows).unwrap()
    } else {
        native_prior(original, binding).unwrap()
    }
}
fn local_chain_fixture() -> (
    antecedent_estimate::joint_design::compiled::JointDesignRecipe,
    antecedent_estimate::joint_design::compiled::JointDesignInference,
    Vec<f64>,
) {
    use antecedent_estimate::joint_design::{
        compiled::{JointDesignInference, JointDesignRecipe},
        local_instrument::{LocalInstrumentDesign, LocalInstrumentUnit},
    };
    let design = LocalInstrumentDesign {
        key: "completeprogram encouragement compliers".into(),
        population: "original encouragement population".into(),
        exclusion: "encouragement has no direct outcome effect".into(),
        monotonicity: "no defiers".into(),
        consistency_and_isolation:
            "consistent complete program and independent encouragement units".into(),
    };
    let units = (0..400)
        .map(|i| {
            let assigned = i % 2 == 0;
            let received = assigned && (i / 2) % 2 == 0;
            LocalInstrumentUnit {
                id: format!("local{i}"),
                assigned,
                received,
                probability: 0.5,
                outcome: if received { if (i / 4) % 2 == 0 { 2.0 } else { 4.0 } } else { 0.0 },
            }
        })
        .collect();
    (
        JointDesignRecipe::InstrumentedLocal { design, units },
        JointDesignInference::LocalComplierBayesian { draws: 512, seed: 445 },
        vec![1.0],
    )
}

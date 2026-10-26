// SPDX-License-Identifier: MIT OR Apache-2.0
//! Public conditional experiment chain, reusing the workflow fixture's real Bayesian fit.
use antecedent::{PreparedStudy, analysis::joint_panel::*};
use antecedent_contract::bridge::*;
use antecedent_contract::evidence_ledger::*;
use antecedent_core::{ExecutionContext, TargetForm};
use antecedent_design::DesignError;
use antecedent_design::decision::{DecisionProblem, Utility};
use antecedent_design::decision_contract::*;
use antecedent_design::scientific_source::{ScientificConsumptionScope, ScientificSourcePayload};
use std::sync::Arc;

#[path = "joint_panel_experiment_chain/matrix.rs"]
mod matrix;

struct ExperimentFixture {
    update: PanelExperimentUpdate,
    ledger: EvidenceLedger,
    coefficients: Vec<f64>,
}
struct DecisionWitness {
    contract: DecisionContract,
    source: PanelExperimentDecisionSource,
}

pub fn exercise(
    prepared: &PreparedStudy<JointPanelState>,
    original: &JointPanelResult,
    ctx: &ExecutionContext,
) {
    let mut fixture = experiment_fixture(original);
    verify_target_refusals(prepared, &fixture, ctx);
    let program = prepared.with_experiment(fixture.update.clone(), &fixture.ledger).unwrap();
    let updated = program.estimate(&fixture.ledger, ctx).unwrap();
    verify_independent_likelihood_reference(original, &updated, &fixture);
    let artifact = verify_experiment_artifacts(prepared, &updated, &fixture, ctx);
    assert_eq!(
        prepared.fits_performed(),
        1,
        "experiment updating must reuse the original fitted mechanism"
    );
    let witness = verify_decision_and_next_study(original, &updated, &fixture, ctx);
    verify_origin_refusals(prepared, &mut fixture, ctx);
    verify_current_authority(&program, &updated, &witness, &artifact, &mut fixture, ctx);
    assert!(
        matches!(
            original.project_law_source(&fixture.update.functional).unwrap().as_numerical_source(),
            ScientificSourcePayload::Law(_)
        ),
        "original panel remains an independent retained result"
    );
    matrix::exercise(prepared, original, ctx);
}

fn verify_independent_likelihood_reference(
    original: &JointPanelResult,
    updated: &PanelExperimentResult,
    fixture: &ExperimentFixture,
) {
    // Independent arithmetic for the declared modular finite-law update. Do not
    // call production conditioning, bridge propagation or decision helpers.
    let prior = original.numerical_project_law_source(&fixture.update.functional).unwrap();
    let posterior = updated.numerical_law_source(&fixture.ledger).unwrap();
    let width = fixture.coefficients.len();
    let experiment = &fixture.update;
    let variance = experiment.observation_variance / experiment.sample_size as f64;
    let means = prior
        .draws()
        .chunks_exact(width)
        .enumerate()
        .map(|(slot, row)| {
            let target = row.iter().zip(&fixture.coefficients).map(|(v, c)| v * c).sum::<f64>();
            (target - experiment.mapping.intercepts[slot]) / experiment.mapping.slopes[slot]
        })
        .collect::<Vec<_>>();
    // Likelihood ratios relative to the best-supported state avoid underflow.
    // Expanding the difference of squares also differs from production's direct
    // log-density construction while retaining the same exact mathematical law.
    let anchor = *means
        .iter()
        .min_by(|a, b| {
            (experiment.observed_mean - **a)
                .abs()
                .total_cmp(&(experiment.observed_mean - **b).abs())
        })
        .unwrap();
    let default_mass = 1.0 / means.len() as f64;
    let unnormalized = means
        .iter()
        .enumerate()
        .map(|(slot, m)| {
            let prior_mass = prior.metadata().weights.as_ref().map_or(default_mass, |w| w[slot]);
            let exponent =
                (m - anchor) * (2.0 * experiment.observed_mean - m - anchor) / (2.0 * variance);
            prior_mass * exponent.exp()
        })
        .collect::<Vec<_>>();
    let normalizer = unnormalized.iter().sum::<f64>();
    let weights = posterior.metadata().weights.as_ref().unwrap();
    for (reference, actual) in unnormalized.iter().zip(weights) {
        assert!((reference / normalizer - actual).abs() < 1e-10);
    }
    let (_, updated_means) = updated.response_means(&fixture.ledger).unwrap();
    for (coordinate, actual) in updated_means.iter().enumerate() {
        let reference = prior
            .draws()
            .chunks_exact(width)
            .zip(&unnormalized)
            .map(|(row, weight)| row[coordinate] * weight / normalizer)
            .sum::<f64>();
        assert!((reference - actual).abs() < 1e-9);
    }
}

fn experiment_fixture(original: &JointPanelResult) -> ExperimentFixture {
    let functional = JointPanelResponseFunctional::TerminalMean;
    let prior=original.numerical_project_law_source(&functional).expect("the real Bayesian fixture must pass its declared posterior diagnostics before experimental updating");
    let width = prior.shape()[1];
    let TargetForm::Contrast(contrast) = original.request().target.form() else {
        panic!("assembled target")
    };
    let mut coefficients = vec![0.0; width];
    for term in contrast.terms() {
        let index =
            original.request().worlds.iter().position(|world| world.world == term.world).unwrap();
        coefficients[index] = term.coefficient;
    }
    let states = prior
        .draws()
        .chunks_exact(width)
        .map(|row| row.iter().zip(&coefficients).map(|(value, weight)| value * weight).sum::<f64>())
        .collect::<Vec<_>>();
    let n = states.len();
    let center = states.iter().sum::<f64>() / n as f64;
    let maximum = states.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(maximum > center + 1e-10, "fixture must retain uncertainty in its whole contrast");
    let slopes = (0..n).map(|slot| if slot % 2 == 0 { 0.95 } else { 1.05 }).collect::<Vec<_>>();
    let slope_mean = slopes.iter().sum::<f64>() / n as f64;
    let slope_variance =
        slopes.iter().map(|value| (value - slope_mean).powi(2)).sum::<f64>() / n as f64;
    let covariance = slopes
        .iter()
        .zip(&states)
        .map(|(slope, value)| (slope - slope_mean) * (value - center))
        .sum::<f64>()
        / n as f64;
    let target = original.experiment_quantity(&functional).unwrap();
    let bridge = declared_bridge(&target, slope_mean, slope_variance, covariance);
    let (ledger, model, mapping, experiment) = fixture_ledger(original);
    let predicted =
        states.iter().zip(&slopes).map(|(value, slope)| value / slope).collect::<Vec<_>>();
    let observed = predicted.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let observation_variance = likelihood_variance(&predicted, 100, 1e-5);
    let update = PanelExperimentUpdate {
        bridge,
        functional: functional.clone(),
        observed_mean: observed,
        observation_variance,
        sample_size: 100,
        modular_policy: PanelExperimentModularPolicy::FrozenJointImportance,
        mapping: PanelExperimentMapping {
            dependency: original.dependency_identity(),
            intercepts: vec![0.0; n],
            slopes,
        },
        evidence: PanelExperimentEvidence { model, prior_origins: vec![], experiment, mapping },
    };
    ExperimentFixture { update, ledger, coefficients }
}

fn fixture_ledger(original: &JointPanelResult) -> (EvidenceLedger, SourceId, SourceId, SourceId) {
    let mut ledger = EvidenceLedger::new();
    let model = ledger
        .add_source(
            SourceDeclaration::new(
                original.dependency_identity().to_hex(),
                SourceKind::FittedModel,
            )
            .with_observations([ObservationId::from_key("panel", b"source")]),
        )
        .unwrap();
    let mapping = ledger
        .add_source(
            SourceDeclaration::new("mapping", SourceKind::FittedModel)
                .with_observations([ObservationId::from_key("mapping", b"source")]),
        )
        .unwrap();
    let experiment = ledger
        .add_source(
            SourceDeclaration::new("geo_experiment", SourceKind::Experiment)
                .with_observations([ObservationId::from_key("geo", b"observed")]),
        )
        .unwrap();
    (ledger, model, mapping, experiment)
}

/// A sampling distribution is not a prior: the frequentist temporal law refuses the Bayesian
/// modular update at both entry points with a typed reason, and its response law is not
/// admitted as a posterior source. The next design is the Bayesian selection posterior on the
/// same temporal frame, which the caller executes through `exercise`.
pub fn exercise_frequentist_boundary(
    prepared: &PreparedStudy<JointPanelState>,
    original: &JointPanelResult,
    ctx: &ExecutionContext,
) {
    let functional = JointPanelResponseFunctional::TerminalMean;
    let n = original.draws.len();
    let target = original.experiment_quantity(&functional).unwrap();
    let bridge = declared_bridge(&target, 1.0, 0.0, 0.0);
    let (ledger, model, mapping, experiment) = fixture_ledger(original);
    let update = PanelExperimentUpdate {
        bridge,
        functional,
        observed_mean: 0.1,
        observation_variance: 1e-3,
        sample_size: 100,
        modular_policy: PanelExperimentModularPolicy::FrozenJointImportance,
        mapping: PanelExperimentMapping {
            dependency: original.dependency_identity(),
            intercepts: vec![0.0; n],
            slopes: vec![1.0; n],
        },
        evidence: PanelExperimentEvidence { model, prior_origins: vec![], experiment, mapping },
    };
    let refused = prepared.with_experiment(update.clone(), &ledger).err().expect("refused");
    assert!(format!("{refused:?}").contains("cannot become a Bayesian experimental prior"));
    assert!(original.condition_experiment(update, &ledger, ctx).is_err());
}

/// A posterior of tens of thousands of slots keeps neighbours of its extreme slot under a very
/// sharp likelihood, so `sharp` is used there. A small panel (a temporal frame fitted in debug
/// time) would be collapsed onto that single slot, leaving no uncertainty for later studies to
/// price: there the standard error of the sample mean is a sixth of the spread of the
/// predicted observations.
fn likelihood_variance(predicted: &[f64], sample_size: u64, sharp: f64) -> f64 {
    if predicted.len() >= 1_000 {
        return sharp;
    }
    let high = predicted.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let low = predicted.iter().copied().fold(f64::INFINITY, f64::min);
    sample_size as f64 * ((high - low) / 6.0).powi(2)
}

fn declared_bridge(
    target: &antecedent_contract::ScientificQuantity,
    slope_mean: f64,
    slope_variance: f64,
    covariance: f64,
) -> BridgeContract {
    let mut experiment_quantity = target.clone();
    experiment_quantity.functional_id = "experimental_sample_mean".into();
    let setting = ExposureSetting {
        population_id: target.population_id.clone(),
        time_start: "program_period:23".into(),
        time_end: "program_period:23".into(),
        dose: 1.0,
        dose_units: "declared_joint_program".into(),
        duration: 1.0,
        duration_units: "panel_period".into(),
        exposure_regime: target.regime_id.clone(),
    };
    let bridge=BridgeContract {
        kind:BridgeKind::ParameterInterventionResponse,
        source:BridgeEndpoint {quantity:experiment_quantity,class:QuantityClass::ModelParameter},
        target:BridgeEndpoint {quantity:target.clone(),class:QuantityClass::InterventionResponse},
        source_value:ValueKind::Mean,target_value:ValueKind::Mean,
        evidence_scope:BridgeScope {population_id:target.population_id.clone(),regime_id:target.regime_id.clone()},
        mapping:BridgeMapping::Affine {intercept:0.0,slope:slope_mean},
        assumptions:vec![BridgeAssumption {id:"declared_experiment_likelihood".into(),kind:AssumptionKind::ParameterInterpretation,statement:"The experimental Gaussian mean maps to this exact future joint-response contrast under frozen modular conditioning".into(),evidence:Some("geo_experiment".into())}],
        transport:None,mapping_uncertainty:Some(MappingUncertainty {id:"paired_mapping".into(),intercept_variance:0.0,slope_variance,intercept_slope_covariance:0.0}),
        source_dependence:Some(SourceMappingDependence::SlopeCovariance {covariance,evidence:"retained_joint_slots".into()}),
        estimand_bridge:Some(EstimandBridge {id:"exact_program_translation".into(),experiment_id:"geo_experiment".into(),experiment:setting.clone(),target:setting,recency_delta:0.0,recency_units:"panel_period".into(),spend_delta:0.0,spend_units:"declared_joint_program".into()}),
    };
    bridge.check_licensing().unwrap();
    bridge
}

fn verify_target_refusals(
    prepared: &PreparedStudy<JointPanelState>,
    fixture: &ExperimentFixture,
    ctx: &ExecutionContext,
) {
    let update = &fixture.update;
    let ledger = &fixture.ledger;
    let mut window_change = update.clone();
    window_change.functional =
        JointPanelResponseFunctional::WindowChange { before: (0, 1), after: (2, 3) };
    assert!(
        prepared.with_experiment(window_change, ledger).is_err(),
        "a one-window experimental bridge cannot authorize a two-window response change"
    );
    let mut wrong_quantity = update.clone();
    wrong_quantity.bridge.target.quantity.functional_id.push_str(":other");
    assert!(
        prepared
            .with_experiment(wrong_quantity, ledger)
            .and_then(|prepared| prepared.estimate(ledger, ctx))
            .is_err(),
        "a licensed bridge for another functional must not update this response"
    );
}

fn verify_experiment_artifacts(
    prepared: &PreparedStudy<JointPanelState>,
    updated: &PanelExperimentResult,
    fixture: &ExperimentFixture,
    ctx: &ExecutionContext,
) -> Vec<u8> {
    let update = &fixture.update;
    let ledger = &fixture.ledger;
    let artifact = updated.export(ledger, ctx).unwrap();
    let historical =
        antecedent_io::joint_panel_experiment_artifact::consume_panel_experiment(&artifact, ctx)
            .unwrap();
    assert!(!historical.grants_current_authority());
    assert_eq!(
        historical.numerical_law().metadata(),
        updated.numerical_law_source(ledger).unwrap().metadata()
    );
    assert!(updated.consume_current(&artifact, ledger, ctx).is_ok());
    assert_fresh_experiment_replay(&artifact, updated.numerical_law_source(ledger).unwrap(), ctx);
    let mut changed_update = update.clone();
    changed_update.observed_mean += 0.5;
    let changed =
        prepared.with_experiment(changed_update, ledger).unwrap().estimate(ledger, ctx).unwrap();
    assert!(changed.consume_current(&artifact, ledger, ctx).is_err());

    artifact
}

#[allow(
    clippy::too_many_lines,
    reason = "a linear test script that checks the decision and then the next study"
)]
fn verify_decision_and_next_study(
    original: &JointPanelResult,
    updated: &PanelExperimentResult,
    fixture: &ExperimentFixture,
    ctx: &ExecutionContext,
) -> DecisionWitness {
    let ledger = &fixture.ledger;
    let coefficients = &fixture.coefficients;
    let functional = &fixture.update.functional;
    let target = original.experiment_quantity(functional).unwrap();
    assert!(!updated.original().identification().is_identified_claim());
    let (before, after) = updated.response_means(ledger).unwrap();
    let dot = |row: &[f64]| {
        row.iter().zip(coefficients).map(|(value, weight)| value * weight).sum::<f64>()
    };
    assert!(
        (dot(after) - dot(before)).abs() > 1e-8,
        "the declared experimental likelihood must change the joint response"
    );
    let threshold = (dot(before) + dot(after)) / 2.0;
    let coordinates = updated.quantities(ledger).unwrap();
    let expression = coefficients.iter().enumerate().fold(
        UtilityExpr::Const(-threshold),
        |expression, (index, coefficient)| {
            UtilityExpr::sum(
                expression,
                UtilityExpr::product(UtilityExpr::Const(*coefficient), UtilityExpr::Input(index)),
            )
        },
    );
    let contract = DecisionContract {
        actions: vec![
            DecisionAction {
                id: "hold".into(),
                kind: ActionKind::Policy,
                inputs: coordinates.clone(),
                utility: UtilityExpr::Input(0),
            },
            DecisionAction {
                id: "change".into(),
                kind: ActionKind::Policy,
                inputs: coordinates.clone(),
                // Both actions retain the same baseline outcome utility; only
                // the coordinated response benefit minus cost changes their ranking.
                utility: UtilityExpr::sum(UtilityExpr::Input(0), expression),
            },
        ],
        utility_units: target.units.clone(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: target.population_id.clone(),
        horizon: coordinates[0].horizon,
        structural_policy: StructuralPolicy::ReportOnly,
    };
    contract.validate().expect("the actual update decision must declare valid response inputs");
    let prior_source = original.project_law_source(functional).unwrap();
    let before_decision =
        prior_source.evaluate(&contract, ScientificConsumptionScope::ModelConditional).unwrap();
    let updated_source = updated.law_source(ledger).unwrap();
    assert!(
        updated_source.authorize(ScientificConsumptionScope::IdentifiedCausal, ledger).is_err(),
        "an experimental importance update must not inherit original causal permission"
    );
    updated_source.authorize(ScientificConsumptionScope::ModelConditional, ledger).unwrap();
    let after_decision = updated.decide(&contract, ledger).unwrap();
    assert_ne!(before_decision.result.verdict, after_decision.result.verdict);
    assert_eq!(after_decision.scope, ScientificConsumptionScope::ModelConditional);
    assert_eq!(after_decision.target_identity, updated_source.target().identity());
    assert_eq!(after_decision.identification_identity, updated_source.identification().identity());
    let problem = DecisionProblem::new(
        vec![0, 1],
        Arc::new(ContrastUtility { coefficients: coefficients.clone(), threshold }),
        vec![],
    );
    let ranking = updated
        .rank_next_samples(
            &problem,
            &[
                PanelFutureSample {
                    id: "cheap".into(),
                    sample_size: 4,
                    observation_variance: 1.0,
                    cost_utility: Some(0.1),
                },
                PanelFutureSample {
                    id: "costly".into(),
                    sample_size: 4,
                    observation_variance: 1.0,
                    cost_utility: Some(0.3),
                },
            ],
            32,
            ledger,
            ctx,
        )
        .unwrap();
    let partly_costed = updated.rank_next_samples(
        &problem,
        &[
            PanelFutureSample {
                id: "costed".into(),
                sample_size: 4,
                observation_variance: 1.0,
                cost_utility: Some(0.1),
            },
            PanelFutureSample {
                id: "uncosted".into(),
                sample_size: 4,
                observation_variance: 1.0,
                cost_utility: None,
            },
        ],
        32,
        ledger,
        ctx,
    );
    assert!(partly_costed.is_err(), "a partly costed catalog must not rank by gross value");
    assert_eq!(ranking[0].id, "cheap");
    assert_eq!(ranking[0].gross_evsi.to_bits(), ranking[1].gross_evsi.to_bits());
    assert!((ranking[0].net_value.unwrap() - (ranking[0].gross_evsi - 0.1)).abs() < 1e-12);
    let (reference, quadrature_error) = reference_next_study_value(updated, fixture, threshold);
    for candidate in &ranking {
        let tolerance = quadrature_error + 6.0 * candidate.standard_error + 1e-9;
        assert!(
            (candidate.gross_evsi - reference).abs() <= tolerance,
            "{}: production EVSI {}, independent reference {reference} +/- {tolerance}",
            candidate.id,
            candidate.gross_evsi
        );
        let cost = if candidate.id == "cheap" { 0.1 } else { 0.3 };
        assert!((candidate.net_value.unwrap() - (reference - cost)).abs() <= tolerance);
    }
    DecisionWitness { contract, source: updated_source }
}

fn reference_next_study_value(
    updated: &PanelExperimentResult,
    fixture: &ExperimentFixture,
    threshold: f64,
) -> (f64, f64) {
    let law = updated.numerical_law_source(&fixture.ledger).unwrap();
    let width = fixture.coefficients.len();
    let mapped = law
        .draws()
        .chunks_exact(width)
        .enumerate()
        .map(|(slot, row)| {
            let target = row.iter().zip(&fixture.coefficients).map(|(v, c)| v * c).sum::<f64>();
            (target - fixture.update.mapping.intercepts[slot]) / fixture.update.mapping.slopes[slot]
        })
        .collect::<Vec<_>>();
    reference_study_value_for_means(updated, fixture, &mapped, threshold)
}

/// Value of a next study whose draw slot `i` is predicted to observe `mapped[i]`.
fn reference_study_value_for_means(
    updated: &PanelExperimentResult,
    fixture: &ExperimentFixture,
    mapped: &[f64],
    threshold: f64,
) -> (f64, f64) {
    // Independent one-dimensional predictive-mixture integration. For the two
    // terminal actions the shared baseline utility cancels, and the integrand is
    // max(0, sum_i w_i Normal(y; mapped_i, variance/n) * (target_i - threshold)).
    // No production signal, update, utility, integration or EVSI helper is called.
    let law = updated.numerical_law_source(&fixture.ledger).unwrap();
    let weights = law.metadata().weights.as_ref().unwrap();
    let width = fixture.coefficients.len();
    let sigma = (1.0_f64 / 4.0).sqrt(); // The independently declared next study above.
    let states = law
        .draws()
        .chunks_exact(width)
        .enumerate()
        .map(|(slot, row)| {
            let target = row.iter().zip(&fixture.coefficients).map(|(v, c)| v * c).sum::<f64>();
            (mapped[slot], weights[slot] * (target - threshold))
        })
        .collect::<Vec<_>>();
    let lower = states.iter().map(|(m, _)| *m).fold(f64::INFINITY, f64::min) - 8.0 * sigma;
    let upper = states.iter().map(|(m, _)| *m).fold(f64::NEG_INFINITY, f64::max) + 8.0 * sigma;
    let step = (upper - lower) / 1024.0;
    let density_scale = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
    let mut future_value = 0.0;
    for cell in 0..1024 {
        let y = lower + (f64::from(cell) + 0.5) * step;
        let advantage = states
            .iter()
            .map(|(m, weighted_gain)| {
                weighted_gain * density_scale * (-0.5 * ((y - m) / sigma).powi(2)).exp()
            })
            .sum::<f64>();
        future_value += step * advantage.max(0.0);
    }
    let current_value = states.iter().map(|(_, gain)| gain).sum::<f64>().max(0.0);
    let absolute_gain = states.iter().map(|(_, gain)| gain.abs()).sum::<f64>();
    // Midpoint error <= h/2 * total variation. Positive-part is Lipschitz,
    // and the L1 norm of a Normal density's derivative is sqrt(2/pi)/sigma.
    let variation_error = step / 2.0 * absolute_gain * (2.0 / std::f64::consts::PI).sqrt() / sigma;
    // The finite integration interval contains +/-8 sigma around every mixture
    // mean. Mills' inequality bounds both omitted tails independently.
    let tail_error =
        absolute_gain * 2.0 * (-32.0_f64).exp() / (8.0 * (2.0 * std::f64::consts::PI).sqrt());
    (future_value - current_value, variation_error + tail_error + 1e-10)
}

fn verify_origin_refusals(
    prepared: &PreparedStudy<JointPanelState>,
    fixture: &mut ExperimentFixture,
    ctx: &ExecutionContext,
) {
    let update = &fixture.update;
    let ledger = &mut fixture.ledger;
    let model = update.evidence.model;
    let experiment = update.evidence.experiment;
    let mut duplicate = update.clone();
    duplicate.evidence.experiment = model;
    assert!(prepared.with_experiment(duplicate, ledger).is_err());
    let prior_from_experiment = ledger
        .add_source(
            SourceDeclaration::new("data_derived_prior", SourceKind::Prior)
                .with_lineage(Lineage::DerivedFromUsedData { inputs: vec![experiment] }),
        )
        .unwrap();
    let mut reused_prior = update.clone();
    reused_prior.evidence.prior_origins.push(prior_from_experiment);
    assert!(prepared.with_experiment(reused_prior, ledger).is_err());
    let mut request = prepared.request().clone();
    request.options.inference =
        antecedent_estimate::joint_panel::PanelInference::Frequentist { draws: 64 };
    request.target =
        request.target.with_inference(antecedent_core::InferenceSelection::Frequentist);
    let frequentist = PreparedStudy::<JointPanelState>::compile_joint_panel(
        prepared.source_data().clone(),
        request,
        ctx,
    )
    .unwrap();
    assert!(
        matches!(frequentist.with_experiment(update.clone(), ledger),
        Err(JointPanelError::Unsupported(reason)) if reason.contains("cluster sampling draws")),
        "a correctly declared frequentist procedure must refuse reinterpretation as an experimental posterior"
    );
    let mut wrong = update.clone();
    wrong.bridge.estimand_bridge = None;
    assert!(prepared.with_experiment(wrong, ledger).is_err());
}

fn verify_current_authority(
    program: &PreparedStudy<PanelExperimentState>,
    updated: &PanelExperimentResult,
    witness: &DecisionWitness,
    artifact: &[u8],
    fixture: &mut ExperimentFixture,
    ctx: &ExecutionContext,
) {
    let ledger = &mut fixture.ledger;
    let model = fixture.update.evidence.model;
    let experiment = fixture.update.evidence.experiment;
    // Unrelated ledger additions leave all actual selected dependencies usable.
    ledger
        .add_source(
            SourceDeclaration::new("unrelated", SourceKind::ObservationSet)
                .with_observations([ObservationId::from_key("other", b"row")]),
        )
        .unwrap();
    assert!(updated.consume_current(&artifact, ledger, ctx).is_ok());
    let mut superseded = ledger.clone();
    superseded
        .supersede_source(
            model,
            SourceDeclaration::new("replacement_model", SourceKind::FittedModel)
                .with_observations([ObservationId::from_key("new_model", b"row")]),
        )
        .unwrap();
    assert!(
        witness
            .source
            .evaluate(&witness.contract, ScientificConsumptionScope::ModelConditional, &superseded)
            .is_err(),
        "a retained source must refuse superseded current model authority"
    );
    ledger.withdraw_source(experiment, "experiment withdrawn").unwrap();
    assert!(updated.response_means(ledger).is_err());
    assert!(
        witness
            .source
            .evaluate(&witness.contract, ScientificConsumptionScope::ModelConditional, ledger)
            .is_err(),
        "a source retained before withdrawal must recheck caller-current authority"
    );
    assert!(
        witness.source.authorize(ScientificConsumptionScope::ModelConditional, ledger).is_err()
    );
    assert!(
        matches!(witness.source.as_numerical_source(), ScientificSourcePayload::Law(_)),
        "historical numerical data remains explicit and usable without scientific authority"
    );
    assert!(updated.export(ledger, ctx).is_err());
    assert!(updated.consume_current(&artifact, ledger, ctx).is_err());
    assert!(
        antecedent_io::joint_panel_experiment_artifact::consume_panel_experiment(&artifact, ctx)
            .is_ok(),
        "historical numerical replay retains its original ledger snapshot without current authority"
    );
    assert!(updated.decide(&witness.contract, ledger).is_err());
    assert!(program.estimate(ledger, ctx).is_err());
}

struct ContrastUtility {
    coefficients: Vec<f64>,
    threshold: f64,
}
impl Utility<usize, PanelExperimentDecisionState> for ContrastUtility {
    fn evaluate_batch(
        &self,
        actions: &[usize],
        states: &[PanelExperimentDecisionState],
        out: &mut [f64],
    ) -> Result<(), DesignError> {
        for (action_index, action) in actions.iter().enumerate() {
            for (slot, state) in states.iter().enumerate() {
                out[action_index * states.len() + slot] = if *action == 0 {
                    0.0
                } else {
                    state
                        .responses
                        .iter()
                        .zip(&self.coefficients)
                        .map(|(value, weight)| value * weight)
                        .sum::<f64>()
                        - self.threshold
                };
            }
        }
        Ok(())
    }
}

fn assert_fresh_experiment_replay(
    bytes: &[u8],
    law: &antecedent_io::distribution_artifact::DistributionArtifact,
    _ctx: &ExecutionContext,
) {
    // Two tests of one process replay concurrently, so each replay owns its directory.
    static REPLAYS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let directory = std::env::temp_dir().join(format!(
        "antecedent-panel-experiment-replay-{}-{}",
        std::process::id(),
        REPLAYS.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("execution.bin"), bytes).unwrap();
    let expected = (
        law.draws().iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        law.metadata()
            .weights
            .as_ref()
            .unwrap()
            .iter()
            .map(|weight| weight.to_bits())
            .collect::<Vec<_>>(),
    );
    std::fs::write(directory.join("expected.json"), serde_json::to_vec(&expected).unwrap())
        .unwrap();
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "joint_panel_workflow::experiment_chain::fresh_process_experiment_artifact_consumer",
            "--nocapture",
        ])
        .env("ANTECEDENT_PANEL_EXPERIMENT_REPLAY", &directory)
        .output()
        .unwrap();
    assert!(
        child.status.success(),
        "fresh experiment replay failed: {}",
        String::from_utf8_lossy(&child.stderr)
    );
    assert!(directory.join("consumer-ran").exists(), "fresh consumer must actually execute");
    std::fs::remove_dir_all(directory).unwrap();
}
#[test]
fn fresh_process_experiment_artifact_consumer() {
    let Some(directory) = std::env::var_os("ANTECEDENT_PANEL_EXPERIMENT_REPLAY") else {
        return;
    };
    let directory = std::path::PathBuf::from(directory);
    let expected: (Vec<u64>, Vec<u64>) =
        serde_json::from_slice(&std::fs::read(directory.join("expected.json")).unwrap()).unwrap();
    let bytes = std::fs::read(directory.join("execution.bin")).unwrap();
    let replay = antecedent_io::joint_panel_experiment_artifact::consume_panel_experiment(
        &bytes,
        &ExecutionContext::for_tests(77),
    )
    .unwrap();
    assert!(!replay.grants_current_authority());
    assert_eq!(
        replay.numerical_law().draws().iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        expected.0
    );
    assert_eq!(
        replay
            .numerical_law()
            .metadata()
            .weights
            .as_ref()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected.1
    );
    std::fs::write(directory.join("consumer-ran"), b"verified").unwrap();
}

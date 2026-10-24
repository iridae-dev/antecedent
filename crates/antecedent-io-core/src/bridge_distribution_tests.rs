//! A bridge of draws against independent references: the delta method and a Monte
//! Carlo of the coefficients drawn here, the Student-t law of a fitted line, the
//! delta method of a measurement correction, the alignment the bridged table keeps,
//! the refusals of what is not a posterior or a predictive law, and the bridge as a
//! stage of a propagation chain.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeScope, MappingUncertainty, QuantityClass, SourceMappingDependence, TrialEffectPair,
    TrialLevelValidation, ValueKind,
};
use antecedent_contract::propagation_chain::{ChainReading, TableConstruction};
use antecedent_contract::{ClaimKind, QuantityRole, ScientificQuantity};
use antecedent_core::ExecutionContext;
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementKind,
    MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
    ParameterEvidence, RepresentationVersion,
};
use antecedent_stats::{normal_ppf, student_t_ppf};

use super::*;
use crate::distribution_artifact::{
    DistributionIdentity, DistributionMetadata, DistributionProvenance,
};
use crate::propagation_chain_record::{
    ChainOpWire, ChainStageWire, ChainUnaryWire, PropagationChainError, PropagationChainRecordWire,
};

const ROWS: usize = 40_000;
const STRIDE: usize = 24_989;

fn quantity_in(
    variable: &str,
    units: &str,
    functional: &str,
    population: &str,
    regime: &str,
) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: population.into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn quantity(variable: &str, units: &str) -> ScientificQuantity {
    quantity_in(variable, units, "average_effect", "adults", "do(t=1)")
}

fn normal_grid(rows: usize, mean: f64, sd: f64) -> Vec<f64> {
    (0..rows).map(|i| mean + sd * normal_ppf((i as f64 + 0.5) / rows as f64)).collect()
}

/// A lattice partner of [`normal_grid`]: draws of an independent normal.
fn partner(rows: usize, mean: f64, sd: f64) -> Vec<f64> {
    let grid = normal_grid(rows, mean, sd);
    (0..rows).map(|i| grid[(i * STRIDE) % rows]).collect()
}

fn table(
    source: &[f64],
    sibling: &[f64],
    semantic: DistributionMeaningWire,
    alignment: DrawAlignment,
) -> DistributionArtifact {
    table_of(
        [quantity("trial:biomarker", "mmol_per_l"), quantity("trial:cost", "usd")],
        source,
        sibling,
        semantic,
        alignment,
    )
}

fn table_of(
    quantities: [ScientificQuantity; 2],
    source: &[f64],
    sibling: &[f64],
    semantic: DistributionMeaningWire,
    alignment: DrawAlignment,
) -> DistributionArtifact {
    let identity = DistributionIdentity::new(
        semantic,
        &quantities,
        alignment,
        DistributionProvenance {
            source_id: "study-1".into(),
            provider_id: "posterior.provider".into(),
            rng_id: "seed:11".into(),
            snapshot_id: "snapshot-1".into(),
            causal_contract_id: "contract-1".into(),
        },
    )
    .expect("identity");
    let mut draws = Vec::with_capacity(source.len() * 2);
    for (a, b) in source.iter().zip(sibling) {
        draws.push(*a);
        draws.push(*b);
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [source.len(), 2],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Unmeasured,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .expect("table")
}

fn posterior_table(source: &[f64]) -> DistributionArtifact {
    table(
        source,
        &partner(source.len(), 0.2, 0.3),
        DistributionMeaningWire::CausalFunctionalPosterior,
        DrawAlignment::Joint,
    )
}

fn validation() -> BridgeAssumption {
    BridgeAssumption {
        id: "validation".into(),
        kind: AssumptionKind::SurrogateValidation,
        statement: "the surrogate is validated at the trial level".into(),
        evidence: Some("meta-analysis:ckd-2024".into()),
    }
}

fn contract(
    mapping: BridgeMapping,
    kind: ValueKind,
    uncertainty: Option<MappingUncertainty>,
    dependence: Option<SourceMappingDependence>,
) -> BridgeContract {
    BridgeContract {
        kind: BridgeKind::SurrogateTerminalOutcome,
        source: BridgeEndpoint {
            quantity: quantity("trial:biomarker", "mmol_per_l"),
            class: QuantityClass::SurrogateEndpoint,
        },
        target: BridgeEndpoint {
            quantity: quantity("trial:survival", "log_hazard"),
            class: QuantityClass::TerminalOutcome,
        },
        source_value: kind,
        target_value: kind,
        evidence_scope: BridgeScope { population_id: "adults".into(), regime_id: "do(t=1)".into() },
        mapping,
        assumptions: vec![validation()],
        transport: None,
        mapping_uncertainty: uncertainty,
        source_dependence: dependence,
        estimand_bridge: None,
    }
}

fn uncertain_affine(covariance: Option<f64>) -> BridgeContract {
    contract(
        BridgeMapping::Affine { intercept: 0.5, slope: 2.0 },
        ValueKind::Posterior,
        Some(MappingUncertainty {
            id: "calibration-fit".into(),
            intercept_variance: 0.01,
            slope_variance: 0.04,
            intercept_slope_covariance: -0.005,
        }),
        Some(match covariance {
            None => SourceMappingDependence::Independent { evidence: "disjoint-samples".into() },
            Some(covariance) => SourceMappingDependence::SlopeCovariance {
                covariance,
                evidence: "shared-sample".into(),
            },
        }),
    )
}

fn moments(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    (mean, values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n)
}

fn near(got: f64, want: f64, tolerance: f64) {
    assert!((got - want).abs() <= tolerance, "{got} vs {want} (tolerance {tolerance})");
}

fn relative(got: f64, want: f64, fraction: f64) {
    near(got, want, fraction * want.abs());
}

/// The delta-method moments of the contract's map of the bridged source, from the
/// propagator the bridge contract owns (on the mean twin of the contract).
fn delta(contract: &BridgeContract, mean: f64, variance: f64) -> (f64, f64) {
    let mut twin = contract.clone();
    twin.source_value = ValueKind::Mean;
    twin.target_value = ValueKind::Mean;
    let propagated = twin.propagate_affine_uncertainty(mean, variance).expect("delta method");
    (propagated.mean, propagated.variance)
}

/// An independent Monte Carlo of `a + b m` over the bridged source column, drawing
/// the coefficients here from their declared joint normal law (and `b` given `m`).
fn reference(source: &[f64], mean: f64, variance: f64, covariance: f64, seed: u64) -> (f64, f64) {
    let (a0, b0, va, vb, d) = (0.5, 2.0, 0.01, 0.04, -0.005);
    let mut rng = CausalRng::from_seed(seed);
    let values: Vec<f64> = source
        .iter()
        .map(|m| {
            let z1 = standard_normal(&mut rng);
            let z2 = standard_normal(&mut rng);
            let vb_given = vb - covariance * covariance / variance;
            let b = b0 + covariance / variance * (m - mean) + vb_given.sqrt() * z1;
            let a = a0 + d / vb_given.sqrt() * z1 + (va - d * d / vb_given).sqrt() * z2;
            a + b * m
        })
        .collect();
    moments(&values)
}

#[test]
fn an_uncertain_slope_agrees_with_the_delta_method_and_a_monte_carlo_reference() {
    let source = normal_grid(ROWS, 1.0, 0.3);
    let bridge = uncertain_affine(None);
    let bridged = bridge_draws(&bridge, &posterior_table(&source), None, 7).expect("bridge");
    let (mean, variance) = moments(&bridged.mapped_column());
    let (delta_mean, delta_variance) =
        delta(&bridge, bridged.source_mean(), bridged.source_variance());
    // Hand computation at mu = 1 and the source's own variance V:
    // 0.5 + 2 = 2.5 and 0.01 + 0.04 + 4 V - 0.01 + 0.04 V = 0.04 + 4.04 V (0.4036 at V = 0.09).
    near(delta_mean, 2.5, 1e-9);
    near(delta_variance, 0.04 + 4.04 * bridged.source_variance(), 1e-9);
    let standard_error = (delta_variance / ROWS as f64).sqrt();
    near(mean, delta_mean, 6.0 * standard_error);
    relative(variance, delta_variance, 0.06);
    let (reference_mean, reference_variance) =
        reference(&source, bridged.source_mean(), bridged.source_variance(), 0.0, 99);
    near(mean, reference_mean, 6.0 * standard_error * 2.0_f64.sqrt());
    relative(variance, reference_variance, 0.06);
    // The exact map adds nothing: the spread is the source's own, 4 * 0.09 = 0.36.
    let exact = contract(
        BridgeMapping::Affine { intercept: 0.5, slope: 2.0 },
        ValueKind::Posterior,
        None,
        None,
    );
    let sharp = bridge_draws(&exact, &posterior_table(&source), None, 7).expect("exact");
    relative(moments(&sharp.mapped_column()).1, 4.0 * bridged.source_variance(), 1e-9);
    assert!(variance > moments(&sharp.mapped_column()).1 + 0.03);
}

#[test]
fn the_declared_dependence_changes_the_bridged_spread_and_mean() {
    let source = normal_grid(ROWS, 1.0, 0.3);
    let independent = bridge_draws(&uncertain_affine(None), &posterior_table(&source), None, 7)
        .expect("independent");
    let covariance = 0.02;
    let bridge = uncertain_affine(Some(covariance));
    let dependent = bridge_draws(&bridge, &posterior_table(&source), None, 7).expect("dependent");
    let (m_independent, v_independent) = moments(&independent.mapped_column());
    let (m_dependent, v_dependent) = moments(&dependent.mapped_column());
    let (delta_mean, delta_variance) =
        delta(&bridge, dependent.source_mean(), dependent.source_variance());
    // The expectation of a product adds the covariance of its factors, and the variance
    // adds 2 mu b c + c^2: 0.08 + 0.0004 here.
    near(delta_mean, 2.5 + covariance, 1e-9);
    near(m_dependent, delta_mean, 6.0 * (delta_variance / ROWS as f64).sqrt());
    relative(v_dependent, delta_variance, 0.06);
    near(m_dependent - m_independent, covariance, 0.012);
    assert!(v_dependent - v_independent > 0.04, "{v_dependent} vs {v_independent}");
    // A covariance no joint law has is refused, not clipped.
    let infeasible = uncertain_affine(Some(0.1));
    let refused = bridge_draws(&infeasible, &posterior_table(&source), None, 7)
        .expect_err("covariance above the product of the variances");
    assert_eq!(refused.detail, "scientific_bridge.dependence_not_a_covariance");
}

#[test]
fn draws_in_draws_out_keep_the_alignment_and_every_other_column() {
    let source = normal_grid(2_000, 1.0, 0.3);
    let input = posterior_table(&source);
    let bridge = uncertain_affine(None);
    let bridged = bridge_draws(&bridge, &input, None, 3).expect("bridge");
    let output = bridged.distribution();
    // The same rows under the same pairing identity, with the bridged column replaced.
    assert_eq!(output.shape(), input.shape());
    assert_eq!(output.alignment_identity().unwrap(), input.alignment_identity().unwrap());
    assert_eq!(bridged.alignment_identity(), input.alignment_identity().unwrap());
    assert_eq!(output.metadata().identity.alignment, DrawAlignment::Joint);
    assert_eq!(output.quantities()[0], ScientificQuantityWire::from(&bridge.target.quantity));
    assert_eq!(output.quantities()[1], input.quantities()[1]);
    let width = 2;
    for row in 0..input.n_draws() {
        assert_eq!(
            output.draws()[row * width + 1].to_bits(),
            input.draws()[row * width + 1].to_bits(),
            "the sibling column is paired row for row"
        );
    }
    assert_eq!(bridged.mapped_column().len(), input.n_draws());
    assert_eq!(bridged.source_column(), source);
    // The bridged table is read as an uncalibrated spread and carries no trust it was not given.
    assert_eq!(output.metadata().calibration, DistributionCalibration::Unmeasured);
    assert_eq!(output.metadata().trust, DistributionTrust::Unverified);

    // An exact map is bit for bit the map of each draw.
    let exact = contract(
        BridgeMapping::Affine { intercept: 0.5, slope: 2.0 },
        ValueKind::Posterior,
        None,
        None,
    );
    let sharp = bridge_draws(&exact, &input, None, 3).expect("exact");
    for (mapped, draw) in sharp.mapped_column().iter().zip(&source) {
        assert_eq!(mapped.to_bits(), (0.5 + 2.0 * draw).to_bits());
    }
    // Replaying with the same seed is bit for bit; another seed is another realization.
    let again = bridge_draws(&bridge, &input, None, 3).expect("again");
    assert_eq!(again.claim().claim.claim_id, bridged.claim().claim.claim_id);
    assert_eq!(again.claim().draws_digest, bridged.claim().draws_digest);
    let other = bridge_draws(&bridge, &input, None, 4).expect("other");
    assert_ne!(other.claim().draws_digest, bridged.claim().draws_digest);
    assert_ne!(other.claim().claim.claim_id, bridged.claim().claim.claim_id);
}

fn trial_contract(kind: ValueKind) -> BridgeContract {
    contract(
        BridgeMapping::TrialLevelLinear(TrialLevelValidation {
            trials: [(0.0, 0.1), (1.0, 1.1), (2.0, 1.9), (3.0, 3.2), (4.0, 3.9)]
                .iter()
                .map(|(surrogate_effect, terminal_effect)| TrialEffectPair {
                    surrogate_effect: *surrogate_effect,
                    terminal_effect: *terminal_effect,
                })
                .collect(),
            min_trials: 5,
            min_r_squared: 0.9,
        }),
        kind,
        None,
        Some(SourceMappingDependence::Independent { evidence: "disjoint-samples".into() }),
    )
}

fn quantile(values: &[f64], p: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() as f64) * p) as usize]
}

#[test]
fn a_trial_line_draws_from_the_student_t_law_of_its_fit_and_a_predictive_law_adds_the_residual() {
    let rows = 60_000;
    let source = vec![2.0; rows];
    // Hand computation: x = 0..4, y = 0.1, 1.1, 1.9, 3.2, 3.9, so the line at x = 2 is the mean 2.04.
    let ybar = 2.04_f64;
    let syy: f64 = [0.1_f64, 1.1, 1.9, 3.2, 3.9].iter().map(|y| (y - ybar).powi(2)).sum();
    let sxy: f64 = [(0.0, 0.1), (1.0, 1.1), (2.0, 1.9), (3.0, 3.2), (4.0, 3.9)]
        .iter()
        .map(|(x, y): &(f64, f64)| (x - 2.0) * (y - ybar))
        .sum();
    let s2 = (syy - sxy * sxy / 10.0) / 3.0;
    let t975 = student_t_ppf(0.975, 3.0);

    let mean_line =
        bridge_draws(&trial_contract(ValueKind::Posterior), &posterior_table(&source), None, 5)
            .expect("posterior");
    let draws = mean_line.mapped_column();
    let scale = (s2 * 0.2).sqrt();
    near(quantile(&draws, 0.5), ybar, 0.05 * scale);
    let spread = quantile(&draws, 0.975) - ybar;
    relative(spread, t975 * scale, 0.07);
    assert!(mean_line.chain_stage("line").is_ok());

    let predictive_table = table(
        &source,
        &partner(rows, 0.2, 0.3),
        DistributionMeaningWire::PosteriorPredictive,
        DrawAlignment::Joint,
    );
    let predictive =
        bridge_draws(&trial_contract(ValueKind::PredictiveLaw), &predictive_table, None, 5)
            .expect("predictive");
    let predicted = predictive.mapped_column();
    let predictive_scale = (s2 * 1.2).sqrt();
    relative(quantile(&predicted, 0.975) - ybar, t975 * predictive_scale, 0.07);
    assert!(quantile(&predicted, 0.975) - ybar > spread);
    // The residual is a column of its own, so a predictive law is a stage of a chain too.
    let stage = predictive.chain_stage("line").expect("a predictive line is a stage");
    assert_eq!(stage.sources.len(), 4);
    assert!(matches!(stage.stage.op, ChainOpWire::Expression { .. }));
    assert_eq!(mean_line.chain_stage("line").expect("stage").sources.len(), 3);
}

#[test]
fn a_trial_line_neither_truncates_nor_extrapolates_a_law() {
    let source = normal_grid(1_000, 2.0, 1.0);
    let refused =
        bridge_draws(&trial_contract(ValueKind::Posterior), &posterior_table(&source), None, 5)
            .expect_err("draws outside the validated range [0, 4]");
    assert_eq!(refused.detail, "scientific_bridge.draws_outside_domain");
    assert_eq!(refused.code, "bridge_mapping_not_licensed");
}

#[test]
fn only_aligned_posterior_or_predictive_draws_about_the_source_are_bridged() {
    let source = normal_grid(500, 1.0, 0.3);
    let bridge = uncertain_affine(None);
    let sibling = partner(500, 0.2, 0.3);
    // An associational claim is never a bridge source.
    let refused =
        bridge_draws(&bridge, &posterior_table(&source), Some(ClaimKind::Associational), 1)
            .expect_err("associational");
    assert_eq!(refused.detail, "scientific_bridge.source_not_draws");
    assert_eq!(refused.supplied.as_deref(), Some("associational"));
    // A bootstrap is neither a posterior nor a predictive law.
    let bootstrap =
        table(&source, &sibling, DistributionMeaningWire::Bootstrap, DrawAlignment::Joint);
    let refused = bridge_draws(&bridge, &bootstrap, None, 1).expect_err("bootstrap");
    assert_eq!(refused.detail, "scientific_bridge.draws_meaning_mismatch");
    // Independent marginals have no row pairing for the bridged draws to keep.
    let marginals = table(
        &source,
        &sibling,
        DistributionMeaningWire::CausalFunctionalPosterior,
        DrawAlignment::IndependentMarginals,
    );
    let refused = bridge_draws(&bridge, &marginals, None, 1).expect_err("marginals");
    assert_eq!(refused.detail, "scientific_bridge.draws_not_aligned");
    // A predictive law is not a posterior.
    let predictive = table(
        &source,
        &sibling,
        DistributionMeaningWire::PosteriorPredictive,
        DrawAlignment::Joint,
    );
    let refused = bridge_draws(&bridge, &predictive, None, 1).expect_err("predictive");
    assert_eq!(refused.detail, "scientific_bridge.value_kind_substitution");
    // A table with no column about the contract's source quantity.
    let mut other = bridge.clone();
    other.source.quantity = quantity("trial:other", "mmol_per_l");
    other.target.quantity = quantity("trial:other_outcome", "log_hazard");
    let refused = bridge_draws(&other, &posterior_table(&source), None, 1).expect_err("quantity");
    assert_eq!(refused.detail, "scientific_bridge.quantity_mismatch");
}

fn clinic_model() -> MeasurementModel {
    MeasurementModel {
        model_id: "screening-flag".into(),
        route: MeasurementRoute::BinaryMisclassification,
        indicator: MeasurementVariable {
            id: "schema:flag".into(),
            kind: MeasurementKind::NoisyMisclassified,
            units: "binary".into(),
        },
        construct: MeasurementVariable {
            id: "construct:disease".into(),
            kind: MeasurementKind::LatentScientific,
            units: "binary".into(),
        },
        binding: MeasurementBinding {
            population_id: "clinic-a".into(),
            regime_id: "observational".into(),
            arm: ArmBinding::AllArms,
        },
        structure: ErrorStructure::NonDifferential,
        parameters: MeasurementParameters::Binary {
            rates: vec![BinaryArmRates {
                arm_id: None,
                sensitivity: 0.9,
                specificity: 0.8,
                sensitivity_variance: 0.0004,
                specificity_variance: 0.0009,
            }],
            min_youden: 0.1,
        },
        representation: RepresentationVersion { coordinate_system_id: "scalar".into(), version: 1 },
        evidence: ParameterEvidence::ValidationSample {
            sample_id: "validation-1".into(),
            units: 200,
        },
    }
}

fn correction_contract() -> BridgeContract {
    let model = clinic_model();
    BridgeContract {
        kind: BridgeKind::LatentMeasuredOutcome,
        source: BridgeEndpoint {
            quantity: quantity_in(
                "schema:flag",
                "probability",
                "prevalence",
                "clinic-a",
                "observational",
            ),
            class: QuantityClass::MeasuredOutcome,
        },
        target: BridgeEndpoint {
            quantity: quantity_in(
                "construct:disease",
                "probability",
                "prevalence",
                "clinic-a",
                "observational",
            ),
            class: QuantityClass::LatentConstruct,
        },
        source_value: ValueKind::Posterior,
        target_value: ValueKind::Posterior,
        evidence_scope: BridgeScope {
            population_id: "clinic-a".into(),
            regime_id: "observational".into(),
        },
        assumptions: vec![BridgeAssumption::for_measurement_model("flag-model", &model)],
        mapping: BridgeMapping::MeasurementCorrection(Box::new(model)),
        transport: None,
        mapping_uncertainty: None,
        source_dependence: Some(SourceMappingDependence::Independent {
            evidence: "validation-sample-disjoint".into(),
        }),
        estimand_bridge: None,
    }
}

#[test]
fn a_measurement_correction_draws_its_rates_and_agrees_with_the_delta_method() {
    let source = normal_grid(ROWS, 0.31, 0.01);
    let input = table_of(
        [
            quantity_in("schema:flag", "probability", "prevalence", "clinic-a", "observational"),
            quantity_in("schema:cost", "usd", "average_effect", "clinic-a", "observational"),
        ],
        &source,
        &partner(ROWS, 0.2, 0.3),
        DistributionMeaningWire::CausalFunctionalPosterior,
        DrawAlignment::Joint,
    );
    let bridged = bridge_draws(&correction_contract(), &input, None, 13).expect("correction");
    let (mean, variance) = moments(&bridged.mapped_column());
    // 0.31 recorded at sensitivity 0.9 and specificity 0.8 is 11/70 latent.
    let truth = 11.0 / 70.0;
    near(mean, truth, 0.004);
    // Delta method: Var(p) = (V(q) + p^2 Var(Se) + (1 - p)^2 Var(Sp)) / J^2, with J = 0.7.
    let delta = (bridged.source_variance()
        + truth * truth * 0.0004
        + (1.0 - truth) * (1.0 - truth) * 0.0009)
        / 0.49;
    relative(variance, delta, 0.1);
    // The rates' spread is in the result: exact rates would leave only the source's.
    assert!(variance > bridged.source_variance() / 0.49 + 5e-4);
    assert_eq!(bridged.mapped_column().len(), ROWS);
    // A stage of a chain: a ratio of the source and the two rate columns.
    let stage = bridged.chain_stage("correction").expect("a correction is a stage");
    assert_eq!(stage.sources.len(), 3);
    assert!(matches!(stage.stage.op, ChainOpWire::Expression { .. }));
    // A recorded prevalence no admissible rates produce is refused, not clipped.
    let impossible = normal_grid(1_000, 0.97, 0.005);
    let input = table_of(
        [
            quantity_in("schema:flag", "probability", "prevalence", "clinic-a", "observational"),
            quantity_in("schema:cost", "usd", "average_effect", "clinic-a", "observational"),
        ],
        &impossible,
        &partner(1_000, 0.2, 0.3),
        DistributionMeaningWire::CausalFunctionalPosterior,
        DrawAlignment::Joint,
    );
    let refused = bridge_draws(&correction_contract(), &input, None, 13).expect_err("unattainable");
    assert_eq!(refused.detail, "scientific_bridge.draws_outside_domain");
}

#[test]
fn a_bridge_is_a_measurement_mapping_stage_and_a_chain_propagates_it_jointly() {
    let source = normal_grid(4_001, 1.0, 0.3);
    let bridge = uncertain_affine(Some(0.02));
    let bridged = bridge_draws(&bridge, &posterior_table(&source), None, 21).expect("bridge");
    let parts = bridged.chain_stage("terminal").expect("an affine bridge is a stage");
    assert_eq!(parts.sources.len(), 3);
    let mut sources = parts.sources.clone();
    // A utility that reads the bridged stage: a nonlinear stage needs the joint law of the draws.
    let stages = vec![
        parts.stage.clone(),
        ChainStageWire::new(
            "utility",
            StageKind::DecisionUtility,
            &["terminal"],
            ChainOpWire::unary(ChainUnaryWire::exponential_utility(1.0)),
        ),
    ];
    let draft = PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        ChainReading::Spread,
        "utility",
        sources.clone(),
        stages.clone(),
    );
    let record = PropagationChainRecordWire::seal(draft, &ExecutionContext::for_tests(17))
        .expect("the chain is licensed");
    let stage = record.results.iter().find(|r| r.stage == "terminal").expect("stage");
    let utility = record.results.iter().find(|r| r.stage == "utility").expect("utility");

    // The stage reproduces the bridged claim: same rows, same coefficients.
    let rows = source.len() as f64;
    near(stage.value("mean").unwrap(), bridged.claim().mean, 1e-9);
    relative(
        stage.value("variance").unwrap(),
        bridged.claim().variance * rows / (rows - 1.0),
        1e-9,
    );
    // The utility is the mean of the utility of each joint draw, not the utility of the mean.
    let mapped = bridged.mapped_column();
    let expectation: f64 = mapped.iter().map(|y| 1.0 - (-y).exp()).sum::<f64>() / rows;
    near(utility.value("mean").unwrap(), expectation, 1e-9);
    let of_the_mean = 1.0 - (-bridged.claim().mean).exp();
    assert!((expectation - of_the_mean).abs() > 1e-3, "{expectation} vs {of_the_mean}");

    // The declared dependence reaches the chain as a shared factor: the stage reads sources
    // that share the covariance's evidence.
    let plan = record.plan().expect("plan");
    let stage_plan = plan.stage("terminal").expect("stage plan");
    assert_eq!(stage_plan.sources.len(), 3);
    assert_eq!(
        stage_plan.dependence.as_ref().map(|d| d.status),
        Some(antecedent_contract::dependence::DependenceStatus::Dependent)
    );

    // An independent bridge brings no shared dependence factor with the source.
    let independent = bridge_draws(&uncertain_affine(None), &posterior_table(&source), None, 21)
        .expect("independent");
    sources = independent.chain_stage("terminal").expect("stage").sources;
    assert!(sources[0].factors.is_empty());
    assert!(!sources[1].factors.is_empty());
}

/// The sample mean and unbiased variance of a column.
fn sample_moments(values: &[f64]) -> (f64, f64) {
    let summary = antecedent_stats::propagation_chain::summarize(values, None).expect("summary");
    (summary.mean, summary.variance)
}

/// Seal a chain that runs `stage` on `sources` and reads it as a spread.
fn seal_stage(
    sources: Vec<ChainSourceWire>,
    stage: ChainStageWire,
) -> Result<PropagationChainRecordWire, PropagationChainError> {
    let output = stage.id.clone();
    let draft = PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        ChainReading::Spread,
        &output,
        sources,
        vec![stage],
    );
    PropagationChainRecordWire::seal(draft, &ExecutionContext::for_tests(17))
}

/// The detail of a typed chain refusal.
fn refusal_detail(error: PropagationChainError) -> &'static str {
    match error {
        PropagationChainError::Refused(refusal) => refusal.detail(),
        other => panic!("a typed refusal, got {other:?}"),
    }
}

fn correction_table(source: &[f64], quantity_id: &str) -> DistributionArtifact {
    table_of(
        [
            quantity_in(quantity_id, "probability", "prevalence", "clinic-a", "observational"),
            quantity_in("schema:cost", "usd", "average_effect", "clinic-a", "observational"),
        ],
        source,
        &partner(source.len(), 0.2, 0.3),
        DistributionMeaningWire::CausalFunctionalPosterior,
        DrawAlignment::Joint,
    )
}

#[test]
fn a_correction_chain_reproduces_the_direct_draws_bridge_bit_for_bit_and_replays() {
    let source = normal_grid(ROWS, 0.31, 0.01);
    let bridged =
        bridge_draws(&correction_contract(), &correction_table(&source, "schema:flag"), None, 13)
            .expect("correction");
    let parts = bridged.chain_stage("correction").expect("a correction is a stage");
    let record = seal_stage(parts.sources.clone(), parts.stage.clone()).expect("licensed");
    let stage = record.results.iter().find(|r| r.stage == "correction").expect("stage");
    // The same inputs give exactly the same rows: the stage's moments are the draws' moments.
    let (mean, variance) = sample_moments(&bridged.mapped_column());
    assert_eq!(stage.value("mean").unwrap().to_bits(), mean.to_bits());
    assert_eq!(stage.value("variance").unwrap().to_bits(), variance.to_bits());
    // The rates are the ones the rows were corrected with, and the chain says so.
    let columns = bridged.correction_columns().expect("rates");
    assert_eq!(columns.sensitivities.len(), ROWS);
    for (row, mapped) in bridged.mapped_column().iter().enumerate().step_by(997) {
        let by_hand = (source[row] + columns.specificities[row] - 1.0)
            / (columns.sensitivities[row] + columns.specificities[row] - 1.0);
        assert_eq!(mapped.to_bits(), by_hand.to_bits());
    }
    // A bridge of another realization of the rates agrees within Monte Carlo error.
    let other =
        bridge_draws(&correction_contract(), &correction_table(&source, "schema:flag"), None, 14)
            .expect("another realization");
    let (other_mean, other_variance) = sample_moments(&other.mapped_column());
    near(other_mean, mean, 6.0 * (2.0 * variance / ROWS as f64).sqrt());
    relative(other_variance, variance, 0.1);
    // The shared factors are declared: the rates share the contract's evidence with the
    // source through the attainability conditioning, and the stage brings its contract.
    let plan = record.plan().expect("plan");
    let stage_plan = plan.stage("correction").expect("stage plan");
    assert_eq!(stage_plan.sources.len(), 3);
    assert_eq!(
        stage_plan.dependence.as_ref().map(|d| d.status),
        Some(antecedent_contract::dependence::DependenceStatus::Dependent)
    );
    assert!(parts.sources.iter().all(|s| s.factors.iter().any(|f| f.kind == "model")));
    assert!(parts.sources[1].factors.iter().any(|f| f.kind == "parameter"));
    // The consumer replays the sealed record to the stored bits, and an exported record decodes.
    let bytes = record.export().expect("exports");
    let decoded = PropagationChainRecordWire::decode(&bytes).expect("decodes");
    let consumed = decoded
        .consume(Some(&record.identity), &ExecutionContext::for_tests(17))
        .expect("the consumer replays it");
    assert_eq!(consumed.results, record.results);
}

#[test]
fn a_correction_chain_feeds_a_utility_the_joint_law_of_the_corrected_rate() {
    let source = normal_grid(4_001, 0.31, 0.01);
    let bridged =
        bridge_draws(&correction_contract(), &correction_table(&source, "schema:flag"), None, 13)
            .expect("correction");
    let parts = bridged.chain_stage("correction").expect("stage");
    let stages = vec![
        parts.stage.clone(),
        ChainStageWire::new(
            "utility",
            StageKind::DecisionUtility,
            &["correction"],
            ChainOpWire::unary(ChainUnaryWire::exponential_utility(8.0)),
        ),
    ];
    let draft = PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        ChainReading::Spread,
        "utility",
        parts.sources.clone(),
        stages,
    );
    let record = PropagationChainRecordWire::seal(draft, &ExecutionContext::for_tests(17))
        .expect("the chain is licensed");
    let utility = record.results.iter().find(|r| r.stage == "utility").expect("utility");
    let rows = source.len() as f64;
    let expectation: f64 =
        bridged.mapped_column().iter().map(|p| 1.0 - (-8.0 * p).exp()).sum::<f64>() / rows;
    near(utility.value("mean").unwrap(), expectation, 1e-9);
}

#[test]
fn a_denominator_that_can_be_zero_or_a_rate_outside_its_range_is_refused_by_the_chain() {
    let source = normal_grid(1_000, 0.31, 0.01);
    let bridged =
        bridge_draws(&correction_contract(), &correction_table(&source, "schema:flag"), None, 13)
            .expect("correction");
    let parts = bridged.chain_stage("correction").expect("stage");
    // The same stage over a sensitivity that, on some rows, leaves the Youden index at or
    // below zero: the ratio is not determined by the draws.
    let mut crossing = parts.sources.clone();
    for (row, bits) in crossing[1].values.iter_mut().enumerate() {
        if row % 7 == 0 {
            *bits = 0.1_f64.to_bits();
        }
    }
    let refused =
        seal_stage(crossing, parts.stage.clone()).expect_err("a denominator crossing zero");
    assert_eq!(refusal_detail(refused), "propagation_chain.denominator_not_bounded");
    // A Youden index above zero but below the model's smallest licensed value is refused too.
    let mut thin = parts.sources.clone();
    let specificity: Vec<f64> = thin[2].values.iter().map(|b| f64::from_bits(*b)).collect();
    for (bits, spec) in thin[1].values.iter_mut().zip(&specificity).take(1) {
        *bits = (1.0 - spec + 0.05).to_bits();
    }
    let refused = seal_stage(thin, parts.stage.clone()).expect_err("a thin Youden index");
    assert_eq!(refusal_detail(refused), "propagation_chain.denominator_not_bounded");
    // A recorded rate above the sensitivity corrects to more than one: refused, not clipped.
    let mut high = parts.sources.clone();
    high[0].values[3] = 0.99_f64.to_bits();
    let refused = seal_stage(high, parts.stage.clone()).expect_err("a rate above one");
    assert_eq!(refusal_detail(refused), "propagation_chain.stage_left_domain");
}

#[test]
fn a_latent_to_measured_correction_is_the_forward_map_of_the_rates() {
    let mut contract = correction_contract();
    std::mem::swap(&mut contract.source, &mut contract.target);
    let source = normal_grid(ROWS, 0.15, 0.01);
    let bridged =
        bridge_draws(&contract, &correction_table(&source, "construct:disease"), None, 13)
            .expect("forward correction");
    let parts = bridged.chain_stage("forward").expect("a forward correction is a stage");
    let record = seal_stage(parts.sources.clone(), parts.stage.clone()).expect("licensed");
    let stage = record.results.iter().find(|r| r.stage == "forward").expect("stage");
    let (mean, variance) = sample_moments(&bridged.mapped_column());
    assert_eq!(stage.value("mean").unwrap().to_bits(), mean.to_bits());
    assert_eq!(stage.value("variance").unwrap().to_bits(), variance.to_bits());
    // 0.9 * 0.15 + 0.2 * 0.85 = 0.305.
    near(mean, 0.305, 2e-3);
}

#[test]
fn a_predictive_line_stage_adds_an_independent_residual_to_the_mean_line() {
    let rows = 8_001;
    let source = normal_grid(rows, 2.0, 0.4);
    let predictive_table = table(
        &source,
        &partner(rows, 0.2, 0.3),
        DistributionMeaningWire::PosteriorPredictive,
        DrawAlignment::Joint,
    );
    let bridged =
        bridge_draws(&trial_contract(ValueKind::PredictiveLaw), &predictive_table, None, 5)
            .expect("predictive");
    let parts = bridged.chain_stage("line").expect("a predictive line is a stage");
    assert_eq!(parts.sources.len(), 4);
    // The residual declares a disturbance of its own, and no other input does.
    let has = |source: &ChainSourceWire, kind: &str| source.factors.iter().any(|f| f.kind == kind);
    assert!(has(&parts.sources[3], "disturbance"));
    assert!(parts.sources[..3].iter().all(|s| !has(s, "disturbance")));

    let record = seal_stage(parts.sources.clone(), parts.stage.clone()).expect("licensed");
    let stage = record.results.iter().find(|r| r.stage == "line").expect("stage");
    let (mean, variance) = sample_moments(&bridged.mapped_column());
    // The same inputs give exactly the bridged draws.
    assert_eq!(stage.value("mean").unwrap().to_bits(), mean.to_bits());
    assert_eq!(stage.value("variance").unwrap().to_bits(), variance.to_bits());

    // The mean line of the same coefficients: the stage without the residual column.
    let mean_stage = ChainStageWire::new(
        "mean_line",
        StageKind::MeasurementMapping,
        &["line.source", "line.slope", "line.intercept"],
        ChainOpWire::affine(ChainCoefficientWire::input(1), ChainCoefficientWire::input(2)),
    );
    let mean_record =
        seal_stage(parts.sources[..3].to_vec(), mean_stage).expect("the mean line is licensed");
    let line = mean_record.results.iter().find(|r| r.stage == "mean_line").expect("mean line");
    let line_variance = line.value("variance").unwrap();
    let residuals = bridged.affine_columns().and_then(|a| a.residuals.clone()).expect("residuals");
    let (_, residual_variance) = sample_moments(&residuals);
    let mean_line: Vec<f64> = {
        let affine = bridged.affine_columns().expect("coefficients");
        (0..rows).map(|i| affine.intercepts[i] + affine.slopes[i] * source[i]).collect()
    };
    let (line_mean, _) = sample_moments(&mean_line);
    let covariance =
        mean_line.iter().zip(&residuals).map(|(a, b)| (a - line_mean) * b).sum::<f64>()
            / (rows as f64 - 1.0);
    // Var(line + residual) = Var(line) + Var(residual) + 2 Cov(line, residual): the residual
    // variance is added, and its covariance with the line is the sampling noise of independence.
    near(variance, line_variance + residual_variance + 2.0 * covariance, 1e-9);
    assert!(residual_variance > 0.01, "{residual_variance}");
}

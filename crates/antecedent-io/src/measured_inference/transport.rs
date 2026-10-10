//! Checked runtime facts of the frozen Gaussian validation protocol.
//! Gaussian/IID model correctness remains a declaration, not a deduction from rows.
use crate::IoError;
use crate::joint_bayesian_transport_artifact::{
    GaussianPriorWire, JointSourceWire, JointTargetWire, PriorProvenanceWire,
};

pub(crate) struct Model<'a> {
    pub(crate) graph: &'a str,
    pub(crate) features: &'a [u32],
    pub(crate) varying: &'a str,
    pub(crate) sharing: &'a str,
    pub(crate) dependence: &'a str,
    pub(crate) invariant: &'a GaussianPriorWire,
    pub(crate) varying_prior: &'a GaussianPriorWire,
    pub(crate) unsupported: f64,
    pub(crate) conflict: f64,
    pub(crate) degree: Option<usize>,
}
fn refused(message: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("cell_not_licensed"),
        message: format!("measured_transport.protocol_not_measured: {message}"),
    }
}
fn prior_matches(prior: &GaussianPriorWire, width: usize) -> bool {
    prior.provenance == PriorProvenanceWire::Declared
        && prior.mean.len() == width
        && prior.mean.iter().all(|v| v.to_bits() == 0.0f64.to_bits())
        && prior.covariance.len() == width * width
        && prior.covariance.iter().enumerate().all(|(i, v)| {
            v.to_bits()
                == if i / width == i % width { 1000.0f64.to_bits() } else { 0.0f64.to_bits() }
        })
}
pub(crate) fn protocol(
    model: &Model<'_>,
    sources: &[JointSourceWire],
    target: &JointTargetWire,
    draws: usize,
    level: f64,
) -> Result<serde_json::Value, IoError> {
    if level.to_bits() != 0.95f64.to_bits()
        || draws < 4096
        || model.graph != "fixed_dag"
        || model.features != [1]
        || model.dependence != "independent_samples"
        || model.unsupported.to_bits() != 0.0f64.to_bits()
        || model.conflict.to_bits() != 3.0f64.to_bits()
        || sources.is_empty()
        || sources.len() > 2
        || model.degree.is_some_and(|d| d != 2)
    {
        return Err(refused(
            "nominal level, posterior draws, model, dependence or source count differs from the frozen validation protocol",
        ));
    }
    let varying_covariates = match model.varying {
        "intercept" => false,
        "intercept_and_covariates" => true,
        _ => return Err(refused("varying block is not measured")),
    };
    let sharing = model.sharing;
    if !((sources.len() == 1 && !varying_covariates && sharing == "independent_varying_blocks")
        || (sources.len() == 2
            && !varying_covariates
            && matches!(sharing, "independent_varying_blocks" | "shared_varying_block"))
        || (sources.len() == 2 && varying_covariates && sharing == "independent_varying_blocks"))
    {
        return Err(refused("varying-block/sharing configuration is not measured"));
    }
    let (q, r) = match (model.degree, varying_covariates) {
        (None, false) => (3, 1),
        (None, true) => (2, 2),
        (Some(2), false) => (5, 1),
        (Some(2), true) => (3, 3),
        _ => return Err(refused("basis is not measured")),
    };
    if !prior_matches(model.invariant, q) || !prior_matches(model.varying_prior, r) {
        return Err(refused(
            "only zero-mean declared isotropic Gaussian priors with variance 1000 were measured; prior banks are not activated",
        ));
    }
    let n = sources[0].outcome.len();
    if !(150..=600).contains(&n)
        || sources.iter().enumerate().any(|(i, s)| {
            s.outcome.len() != n
                || s.covariates.len() != 1
                || s.covariates[0].len() != n
                || s.covariates[0].iter().any(|x| !x.is_finite() || !(-1.0..=1.0).contains(x))
                || s.noise_variance.to_bits()
                    != if i == 0 { 1.0f64.to_bits() } else { 2.25f64.to_bits() }
        })
    {
        return Err(refused(
            "balanced per-source sample sizes, known variances or covariate bounds differ from the frozen protocol",
        ));
    }
    let target_design = [-0.25f64, 0.25, 0.25, 0.55];
    if target.rows != 4
        || target.covariates.len() != 1
        || target.covariates[0].len() != 4
        || target.covariates[0].iter().zip(target_design).any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(refused("target covariate design differs from the measured fixed target"));
    }
    Ok(serde_json::json!({
        "validation_design":if model.degree.is_some(){"known_quadratic_gaussian_invariant_effect_fixed_target"}else{"known_gaussian_invariant_effect_fixed_target"},
        "evidence_interpretation":"finite fixed-parameter validation grid; no universal prior or arbitrary-model coverage claim",
        "declared_model_assumptions":["correct invariant Gaussian effect model","independent disjoint source samples","known residual variances"],
        "sampling_assumptions_authenticated_from_rows":false,
        "validation_design_source_covariates":"uniform(-1,1)","checked_source_covariate_bounds":[-1.0,1.0],
        "validation_design_treatment_probabilities":if sources.len()==1{vec![0.5]}else{vec![0.5,0.35]},
        "validation_design_invariant_effect_coefficients":if model.degree.is_some(){vec![2.0,0.4,0.2]}else{vec![2.0,0.4]},
        "source_rows":sources.iter().map(|s|s.outcome.len()).collect::<Vec<_>>(),
        "source_ids":sources.iter().map(|s|&s.id).collect::<Vec<_>>(),"source_noise_variances":sources.iter().map(|s|s.noise_variance).collect::<Vec<_>>(),
        "target_covariates":target.covariates,"features":model.features,"basis_degree":model.degree,
        "varying":model.varying,"sharing":model.sharing,"prior_mean":0.0,"prior_variance":1000.0,"prior_provenance":"declared",
        "posterior_draws":draws,"posterior_draws_min":4096,"level":level,"reported_scalar":"target_effect"
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint_bayesian_transport_artifact::DataIdentityWire;

    fn prior(width: usize) -> GaussianPriorWire {
        GaussianPriorWire {
            mean: vec![0.0; width],
            covariance: (0..width * width)
                .map(|i| if i / width == i % width { 1000.0 } else { 0.0 })
                .collect(),
            provenance: PriorProvenanceWire::Declared,
        }
    }
    fn inputs() -> (Vec<JointSourceWire>, JointTargetWire) {
        let identity = DataIdentityWire { snapshot_digest: "source".into(), datum_ids: vec![] };
        let sources = vec![JointSourceWire {
            id: "s0".into(),
            identity: identity.clone(),
            treatment: vec![true; 150],
            outcome: vec![0.0; 150],
            covariates: vec![vec![0.0; 150]],
            noise_variance: 1.0,
        }];
        let target =
            JointTargetWire { identity, rows: 4, covariates: vec![vec![-0.25, 0.25, 0.25, 0.55]] };
        (sources, target)
    }
    #[test]
    fn frozen_protocol_checks_actual_inputs_and_labels_assumptions_honestly() {
        let invariant = prior(3);
        let varying_prior = prior(1);
        let mut model = Model {
            graph: "fixed_dag",
            features: &[1],
            varying: "intercept",
            sharing: "independent_varying_blocks",
            dependence: "independent_samples",
            invariant: &invariant,
            varying_prior: &varying_prior,
            unsupported: 0.0,
            conflict: 3.0,
            degree: None,
        };
        let (mut sources, mut target) = inputs();
        let report = protocol(&model, &sources, &target, 4096, 0.95).unwrap();
        assert_eq!(report["sampling_assumptions_authenticated_from_rows"], false);
        assert!(protocol(&model, &sources, &target, 4095, 0.95).is_err());
        assert!(protocol(&model, &sources, &target, 4096, 0.90).is_err());
        sources[0].noise_variance = 1.1;
        assert!(protocol(&model, &sources, &target, 4096, 0.95).is_err());
        sources[0].noise_variance = 1.0;
        sources[0].covariates[0][0] = 1.1;
        assert!(protocol(&model, &sources, &target, 4096, 0.95).is_err());
        sources[0].covariates[0][0] = 0.0;
        target.covariates[0][0] = -0.20;
        assert!(protocol(&model, &sources, &target, 4096, 0.95).is_err());
        target.covariates[0][0] = -0.25;
        model.degree = Some(3);
        assert!(protocol(&model, &sources, &target, 4096, 0.95).is_err());
        model.degree = None;
        let bank = GaussianPriorWire {
            provenance: PriorProvenanceWire::Bank { bank_id: "bank".into(), consumed: vec![] },
            ..prior(3)
        };
        model.invariant = &bank;
        assert!(protocol(&model, &sources, &target, 4096, 0.95).is_err());
        model.invariant = &invariant;
        sources[0].outcome.truncate(149);
        assert!(protocol(&model, &sources, &target, 4096, 0.95).is_err());
    }
}

#[cfg(test)]
mod native_lifecycle_tests {
    use antecedent_core::{ExecutionContext, VariableId};
    use antecedent_estimate::joint_bayesian_transport::*;
    use antecedent_estimate::learned_joint_transport::{
        LearnedJointModel, fit_learned_joint_transport,
    };
    use antecedent_identify::{
        PopulationFactor, TransportCertificate, TransportFormula, TransportIdentification,
    };
    use std::sync::Arc;

    #[test]
    #[allow(
        clippy::cast_precision_loss,
        clippy::too_many_lines,
        reason = "bounded paired native execution and adversarial replay fixture"
    )]
    fn every_measured_transport_variant_matches_original_execution_basis_bit_for_bit() {
        let factor = PopulationFactor {
            population: Arc::from("source"),
            regime: None,
            variables: Arc::from([]),
            conditioned_on: Arc::from([]),
            interventions: Arc::from([]),
        };
        let proof = TransportIdentification::Transportable {
            formula: TransportFormula::Standardize {
                over: Arc::from([VariableId::from_raw(1)]),
                source_response: factor.clone(),
                target_law: factor,
            },
            certificate: TransportCertificate {
                rule: Arc::from("standardization"),
                selection_targets: Arc::from([]),
                premises: Arc::from([]),
            },
        };
        let target = TargetData {
            identity: DataIdentity {
                snapshot_digest: "target".into(),
                datum_ids: (0..4).map(|i| format!("t{i}")).collect(),
            },
            rows: 4,
            covariates: vec![vec![-0.25, 0.25, 0.25, 0.55]],
        };
        let options = JointTransportOptions { draws: 4096, seed: 701 };
        let ctx = ExecutionContext::for_tests(19);
        for (count, varying, sharing) in [
            (1, VaryingBlock::Intercept, SourceSharing::IndependentVaryingBlocks),
            (2, VaryingBlock::Intercept, SourceSharing::IndependentVaryingBlocks),
            (2, VaryingBlock::Intercept, SourceSharing::SharedVaryingBlock),
            (2, VaryingBlock::InterceptAndCovariates, SourceSharing::IndependentVaryingBlocks),
        ] {
            let sources: Vec<_> = (0..count)
                .map(|b| {
                    let x: Vec<_> = (0..150)
                        .map(|i| (f64::from(i) * 0.71 + f64::from(b) * 0.2).sin())
                        .collect();
                    let a: Vec<_> = (0..150).map(|i| i % 3 != 0).collect();
                    let y: Vec<_> = x
                        .iter()
                        .zip(&a)
                        .enumerate()
                        .map(|(i, (x, a))| {
                            0.4 + 0.7 * f64::from(b)
                                + 0.5 * x
                                + if *a { 2.0 + 0.4 * x } else { 0.0 }
                                + 0.1 * (i as f64 * 1.37).cos()
                        })
                        .collect();
                    SourceData {
                        id: format!("s{b}"),
                        identity: DataIdentity {
                            snapshot_digest: format!("s{b}"),
                            datum_ids: (0..150).map(|i| format!("s{b}-{i}")).collect(),
                        },
                        treatment: a,
                        outcome: y,
                        covariates: vec![x],
                        noise_variance: if b == 0 { 1.0 } else { 2.25 },
                    }
                })
                .collect();
            let varying_cov = varying == VaryingBlock::InterceptAndCovariates;
            let model = JointTransportModel {
                graph: TransportGraphClass::FixedDag,
                features: vec![1],
                varying,
                sharing,
                dependence: SourceDependence::IndependentSamples,
                priors: JointPriors {
                    invariant: GaussianPrior::isotropic(
                        if varying_cov { 2 } else { 3 },
                        0.0,
                        1000.0,
                        PriorProvenance::Declared,
                    ),
                    varying: GaussianPrior::isotropic(
                        if varying_cov { 2 } else { 1 },
                        0.0,
                        1000.0,
                        PriorProvenance::Declared,
                    ),
                },
                max_unsupported_mass: 0.0,
                conflict_z_threshold: 3.0,
            };
            let joint = fit_joint_bayesian_transport(
                &proof,
                &model,
                &sources,
                Some(&target),
                &options,
                &ctx,
            )
            .unwrap();
            assert_eq!(
                joint.target_calibration_basis(0.95).unwrap().estimator.as_ref(),
                "joint_bayesian_transport"
            );
            let (wire, _, measured) = crate::joint_bayesian_transport_artifact::JointBayesianArtifactWire::build_measured(&proof,&model,&sources,&target,options,0.95,&ctx).unwrap();
            assert_eq!(wire.export().unwrap(), measured.source_artifact());
            assert_eq!(
                measured.report().scalars[0].point.to_bits(),
                joint.target_effect_mean.to_bits()
            );
            assert_eq!(measured.report().scalars[0].calibration.status, "calibrated");
            let (_,_,loaded)=crate::joint_bayesian_transport_artifact::JointBayesianArtifactWire::consume_measured(measured.export(),measured.expectation(),crate::joint_bayesian_transport_artifact::JointBayesianConsumeLimits::default(),&ctx).unwrap();
            assert_eq!(loaded.export(), measured.export());
            // Canonical harness identities keep the exact pre-change evidence key.
            assert_eq!(
                joint.target_calibration_basis(0.95).unwrap().posterior.as_ref(),
                format!(
                    "{}.{}",
                    antecedent_estimate::joint_bayesian_transport::JOINT_TRANSPORT_SAMPLER,
                    joint.model_identity
                )
            );
            let mut aliases = sources.clone();
            for (index, source) in aliases.iter_mut().enumerate() {
                source.id = format!("renamed-population-{index}");
            }
            let (_, alias_fit, alias_measured) = crate::joint_bayesian_transport_artifact::JointBayesianArtifactWire::build_measured(
                &proof, &model, &aliases, &target, options, 0.95, &ctx).unwrap();
            assert_eq!(
                alias_fit.target_calibration_basis(0.95).unwrap(),
                joint.target_calibration_basis(0.95).unwrap()
            );
            assert_eq!(alias_fit.draws.values, joint.draws.values);
            assert_ne!(alias_fit.model_identity, joint.model_identity);
            assert_ne!(alias_measured.source_artifact(), measured.source_artifact());
            assert_ne!(alias_measured.expectation(), measured.expectation());
            let alias_loaded = crate::measured_inference::MeasuredInference::consume(
                alias_measured.export(),
                alias_measured.expectation(),
                &ctx,
            )
            .unwrap();
            assert_eq!(alias_loaded.export(), alias_measured.export());
            assert!(
                crate::measured_inference::MeasuredInference::consume(
                    alias_measured.export(),
                    measured.expectation(),
                    &ctx
                )
                .is_err()
            );
            let mut forged =
                crate::measured_inference::decode(measured.export(), measured.expectation())
                    .unwrap();
            forged.report.scalars[0].point += 0.1;
            forged.report.identity.seal =
                crate::measured_inference::report_seal(&forged.report).unwrap();
            let forged_bytes = crate::to_cbor(&forged).unwrap();
            assert!(crate::joint_bayesian_transport_artifact::JointBayesianArtifactWire::consume_measured(&forged_bytes,&forged.report.identity,crate::joint_bayesian_transport_artifact::JointBayesianConsumeLimits::default(),&ctx).is_err());

            if count == 1 {
                let generic = crate::measured_inference::MeasuredInference::consume(
                    measured.export(),
                    measured.expectation(),
                    &ctx,
                )
                .unwrap();
                assert_eq!(generic.export(), measured.export());
                let mut wrong_route = measured.expectation().clone();
                wrong_route.route = "invented".into();
                assert!(
                    crate::measured_inference::MeasuredInference::consume(
                        measured.export(),
                        &wrong_route,
                        &ctx
                    )
                    .is_err()
                );
                // Replacing source bytes with a valid newly produced source and
                // resealing the entire envelope still cannot retain old scalars.
                let mut changed_sources = sources.clone();
                changed_sources[0].outcome[0] += 1.0;
                let (changed_wire, _) =
                    crate::joint_bayesian_transport_artifact::JointBayesianArtifactWire::build(
                        &proof,
                        &model,
                        &changed_sources,
                        &target,
                        options,
                        &ctx,
                    )
                    .unwrap();
                let mut swapped =
                    crate::measured_inference::decode(measured.export(), measured.expectation())
                        .unwrap();
                swapped.source = changed_wire.export().unwrap();
                swapped.report.identity.candidate_digest = crate::identity::digest_canonical(
                    antecedent_core::IdentityDomain::Claim,
                    &swapped.source,
                )
                .to_hex();
                swapped.report.identity.premises_digest = changed_wire.premises_digest;
                swapped.report.identity.data_digest = changed_wire.data_digest;
                swapped.report.identity.seal =
                    crate::measured_inference::report_seal(&swapped.report).unwrap();
                let bytes = crate::to_cbor(&swapped).unwrap();
                assert!(crate::joint_bayesian_transport_artifact::JointBayesianArtifactWire::consume_measured(&bytes,&swapped.report.identity,crate::joint_bayesian_transport_artifact::JointBayesianConsumeLimits::default(),&ctx).is_err());
            }
            let learned_model = LearnedJointModel {
                graph: model.graph,
                features: model.features,
                basis_degree: 2,
                varying,
                sharing,
                dependence: model.dependence,
                priors: JointPriors {
                    invariant: GaussianPrior::isotropic(
                        if varying_cov { 3 } else { 5 },
                        0.0,
                        1000.0,
                        PriorProvenance::Declared,
                    ),
                    varying: GaussianPrior::isotropic(
                        if varying_cov { 3 } else { 1 },
                        0.0,
                        1000.0,
                        PriorProvenance::Declared,
                    ),
                },
                max_unsupported_mass: 0.0,
                conflict_z_threshold: 3.0,
            };
            let learned = fit_learned_joint_transport(
                &proof,
                &learned_model,
                &sources,
                Some(&target),
                &options,
                &ctx,
            )
            .unwrap();
            assert_eq!(
                learned.target_calibration_basis(0.95).unwrap().estimator.as_ref(),
                "learned_joint_transport"
            );
            let (wire, _, measured) =
                crate::learned_joint_transport_artifact::LearnedJointArtifactWire::build_measured(
                    &proof,
                    &learned_model,
                    &sources,
                    &target,
                    options,
                    0.95,
                    &ctx,
                )
                .unwrap();
            assert_eq!(wire.export().unwrap(), measured.source_artifact());
            assert_eq!(
                measured.report().scalars[0].point.to_bits(),
                learned.target_effect_mean.to_bits()
            );
            let (_,_,loaded)=crate::learned_joint_transport_artifact::LearnedJointArtifactWire::consume_measured(measured.export(),measured.expectation(),crate::learned_joint_transport_artifact::LearnedJointConsumeLimits::default(),&ctx).unwrap();
            assert_eq!(loaded.export(), measured.export());
            // Canonical harness identities keep the exact pre-change evidence key.
            assert_eq!(
                learned.target_calibration_basis(0.95).unwrap().posterior.as_ref(),
                format!(
                    "{}.{}",
                    antecedent_estimate::learned_joint_transport::LEARNED_JOINT_SAMPLER,
                    learned.model_identity
                )
            );
            let mut aliases = sources.clone();
            for (index, source) in aliases.iter_mut().enumerate() {
                source.id = format!("renamed-population-{index}");
            }
            let (_, alias_fit, alias_measured) =
                crate::learned_joint_transport_artifact::LearnedJointArtifactWire::build_measured(
                    &proof,
                    &learned_model,
                    &aliases,
                    &target,
                    options,
                    0.95,
                    &ctx,
                )
                .unwrap();
            assert_eq!(
                alias_fit.target_calibration_basis(0.95).unwrap(),
                learned.target_calibration_basis(0.95).unwrap()
            );
            assert_eq!(alias_fit.draws.values, learned.draws.values);
            assert_ne!(alias_fit.model_identity, learned.model_identity);
            assert_ne!(alias_measured.source_artifact(), measured.source_artifact());
            assert_ne!(alias_measured.expectation(), measured.expectation());
            let alias_loaded = crate::measured_inference::MeasuredInference::consume(
                alias_measured.export(),
                alias_measured.expectation(),
                &ctx,
            )
            .unwrap();
            assert_eq!(alias_loaded.export(), alias_measured.export());
            assert!(
                crate::measured_inference::MeasuredInference::consume(
                    alias_measured.export(),
                    measured.expectation(),
                    &ctx
                )
                .is_err()
            );
            let mut forged =
                crate::measured_inference::decode(measured.export(), measured.expectation())
                    .unwrap();
            forged.report.scalars[0].calibration.calibration_sha = Some("0".repeat(40));
            forged.report.identity.seal =
                crate::measured_inference::report_seal(&forged.report).unwrap();
            let forged_bytes = crate::to_cbor(&forged).unwrap();
            assert!(crate::learned_joint_transport_artifact::LearnedJointArtifactWire::consume_measured(&forged_bytes,&forged.report.identity,crate::learned_joint_transport_artifact::LearnedJointConsumeLimits::default(),&ctx).is_err());
        }
    }
}

//! Actual nested-model executions licensed only by current method-specific records.
use super::{
    MeasuredExpectation, MeasuredInference, ScalarExecution, authorize, compare_replay, decode,
};
use crate::IoError;
use crate::nested_markov_artifact::{
    NestedFisherArtifact, NestedMarkovConsumeLimits, NestedMarkovExpectation, NestedOptionsWire,
};
use crate::nested_markov_bayesian_artifact::{Artifact, BayesianFunctional, Expectation, Limits};
use antecedent_core::ExecutionContext;
use antecedent_estimate::nested_markov_binary::{FitOptions, NestedMarkovInput};
use antecedent_estimate::nested_markov_uncertainty::{
    NestedFisherFunctional, NestedMarkovUncertainty,
};
use antecedent_learn::nested_markov_bayesian::{Options, Prior};

fn outside(message: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("cell_not_licensed"),
        message: format!("nested_markov.measured_scope: {message}"),
    }
}
fn label_scope(names: &[String]) -> Result<(), IoError> {
    if names.iter().any(|name| name.len() > 256) {
        return Err(outside(
            "original graph variable labels exceed the measured 256 UTF-8 byte bound",
        ));
    }
    Ok(())
}
fn fit_scope(options: &FitOptions) -> Result<(), IoError> {
    let frozen = FitOptions::default();
    if options.max_iterations != frozen.max_iterations
        || options.tolerance.to_bits() != frozen.tolerance.to_bits()
        || options.refuse_constraint_residual_above != frozen.refuse_constraint_residual_above
    {
        return Err(outside(
            "only the measured default constrained-likelihood fitting protocol is licensed",
        ));
    }
    Ok(())
}
fn wire_fit_scope(options: NestedOptionsWire) -> Result<(), IoError> {
    fit_scope(&options.to_options())
}
fn bayesian_scope(prior: &Prior, options: &Options) -> Result<(), IoError> {
    let uniform =
        |shape: f64| prior.alpha.iter().chain(&prior.beta).all(|x| x.to_bits() == shape.to_bits());
    if !(uniform(1.0) || uniform(2.0))
        || options.chains != 4
        || options.warmup != 2048
        || options.draws != 4096
        || options.max_proposals != 5_000_000
        || options.credible_mass.to_bits() != 0.95_f64.to_bits()
    {
        return Err(outside(
            "only uniform eleven-coordinate Beta(1,1) or Beta(2,2) priors and the frozen 4-chain sampler are measured",
        ));
    }
    Ok(())
}
fn fisher_authority(
    artifact: &NestedFisherArtifact,
    actual: &NestedMarkovUncertainty,
) -> Result<MeasuredInference, IoError> {
    wire_fit_scope(artifact.point.options)?;
    if actual.nominal_level.to_bits() != 0.95_f64.to_bits() {
        return Err(outside("only nominal level 0.95 is measured"));
    }
    let points = [
        artifact.point.receipt.contrast.model_means[0],
        artifact.point.receipt.contrast.model_means[1],
        artifact.point.receipt.contrast.model_contrast,
    ];
    let executions = [
        ("mean0", NestedFisherFunctional::Mean0),
        ("mean1", NestedFisherFunctional::Mean1),
        ("contrast", NestedFisherFunctional::Contrast),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (name, functional))| ScalarExecution {
        name: name.into(),
        point: points[i],
        interval: (actual.interval_candidates[i][0], actual.interval_candidates[i][1]),
        basis: actual.calibration_basis(functional),
    })
    .collect();
    authorize(
        "nested_fisher",
        artifact.export()?,
        &artifact.point.premises_digest,
        &artifact.point.data_digest,
        serde_json::json!({"nominal_level":0.95,"max_variable_name_bytes":256,"fit_options":artifact.point.options,"sampling":artifact.sampling,"parameter_covariance_standing":"regular-model approximation; only named scalar intervals are calibrated","model_assumptions":"declared IID multinomial, correctly specified interior Verma nested model; observations do not authenticate the generating law"}),
        executions,
    )
}
fn bayesian_authority(artifact: &Artifact) -> Result<MeasuredInference, IoError> {
    wire_fit_scope(artifact.point.options)?;
    bayesian_scope(&artifact.prior, &artifact.options)?;
    if artifact.posterior.diagnostics.len() != 14
        || artifact.posterior.diagnostics.iter().any(|d| {
            !d.rank_rhat.is_finite()
                || !d.folded_rhat.is_finite()
                || d.rank_rhat.max(d.folded_rhat) > 1.01
                || !d.bulk_ess.is_finite()
                || !d.tail_ess.is_finite()
                || d.bulk_ess < 400.0
                || d.tail_ess < 400.0
        })
    {
        return Err(outside(
            "all fourteen original posterior diagnostics must pass the frozen Rhat/ESS criteria",
        ));
    }
    let executions = [
        ("mean0", BayesianFunctional::Mean0),
        ("mean1", BayesianFunctional::Mean1),
        ("contrast", BayesianFunctional::Contrast),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (name, functional))| ScalarExecution {
        name: name.into(),
        point: artifact.posterior.mean[11 + i],
        interval: (artifact.posterior.credible[11 + i][0], artifact.posterior.credible[11 + i][1]),
        basis: artifact.calibration_basis(functional),
    })
    .collect();
    authorize(
        "nested_bayesian",
        artifact.export()?,
        &artifact.premises_digest,
        &artifact.point.data_digest,
        serde_json::json!({"nominal_level":0.95,"max_variable_name_bytes":256,"fit_options":artifact.point.options,"prior":artifact.prior,"sampler":artifact.options,"diagnostic_scope":"all fourteen coordinates; rank/folded Rhat <=1.01 and bulk/tail ESS >=400","posterior_standing":"Monte Carlo posterior means/covariance/quantile estimates; only named scalar intervals calibrated, not simultaneous or all parameter intervals","model_assumptions":artifact.sampling}),
        executions,
    )
}
impl NestedFisherArtifact {
    /// Execute the original Fisher engine and authorize every reported scalar.
    /// The original candidate retains its unmeasured standing.
    /// # Errors
    /// Unsupported protocol, native estimation refusal, or missing current calibration.
    pub fn build_measured(
        input: &NestedMarkovInput,
        options: &FitOptions,
        level: f64,
        ctx: &ExecutionContext,
    ) -> Result<(Self, MeasuredInference), IoError> {
        label_scope(&input.graph.variables)?;
        fit_scope(options)?;
        if level.to_bits() != 0.95_f64.to_bits() {
            return Err(outside("only nominal level 0.95 is measured"));
        }
        let (artifact, actual) = Self::build_with_uncertainty(input, options, level, ctx)?;
        let measured = fisher_authority(&artifact, &actual)?;
        Ok((artifact, measured))
    }
    /// Independently replay original source and resolve the actual scalar bases.
    /// # Errors
    /// Changed source/question/receipt, native scope refusal, or nonattesting records.
    pub fn consume_measured(
        bytes: &[u8],
        expected: &MeasuredExpectation,
        limits: NestedMarkovConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<(Self, MeasuredInference), IoError> {
        let wire = decode(bytes, expected)?;
        if expected.route != "nested_fisher" || wire.source.len() > 256 * 1024 {
            return Err(outside("Fisher route or source bound mismatch"));
        }
        let declared: Self = crate::from_cbor(&wire.source)?;
        label_scope(&declared.point.graph.variables)?;
        wire_fit_scope(declared.point.options)?;
        if declared.nominal_level.to_bits() != 0.95_f64.to_bits() {
            return Err(outside("only nominal level 0.95 is measured"));
        }
        drop(declared);
        let original = NestedMarkovExpectation {
            premises_digest: Some(expected.premises_digest.clone()),
            data_digest: Some(expected.data_digest.clone()),
        };
        let (artifact, actual) =
            Self::consume_with_uncertainty(&wire.source, &original, limits, ctx)?;
        let measured = fisher_authority(&artifact, &actual)?;
        compare_replay(&wire, &measured)?;
        Ok((artifact, measured))
    }
}
impl Artifact {
    /// Execute the unchanged eleven-dimensional posterior and authorize three scalars.
    /// # Errors
    /// Unsupported prior/fit/sampler, original refusal, or missing current calibration.
    pub fn build_measured(
        input: &NestedMarkovInput,
        fit_options: &FitOptions,
        prior: Prior,
        options: Options,
        ctx: &ExecutionContext,
    ) -> Result<(Self, MeasuredInference), IoError> {
        label_scope(&input.graph.variables)?;
        fit_scope(fit_options)?;
        bayesian_scope(&prior, &options)?;
        let artifact = Self::build(input, fit_options, prior, options, ctx)?;
        let measured = bayesian_authority(&artifact)?;
        Ok((artifact, measured))
    }
    /// Independently replay the checked point and every posterior draw before licensing.
    /// # Errors
    /// Changed source/question/receipt, bounded native refusal, or nonattesting records.
    pub fn consume_measured(
        bytes: &[u8],
        expected: &MeasuredExpectation,
        limits: Limits,
        ctx: &ExecutionContext,
    ) -> Result<(Self, MeasuredInference), IoError> {
        let wire = decode(bytes, expected)?;
        if expected.route != "nested_bayesian"
            || wire.source.len() > limits.bytes.min(16 * 1024 * 1024)
        {
            return Err(outside("Bayesian route or source bound mismatch"));
        }
        let declared: Self = crate::from_cbor(&wire.source)?;
        label_scope(&declared.point.graph.variables)?;
        wire_fit_scope(declared.point.options)?;
        bayesian_scope(&declared.prior, &declared.options)?;
        drop(declared);
        let original = Expectation {
            premises_digest: Some(expected.premises_digest.clone()),
            point: NestedMarkovExpectation {
                premises_digest: None,
                data_digest: Some(expected.data_digest.clone()),
            },
        };
        let artifact = Self::consume(&wire.source, &original, limits, ctx)?;
        let measured = bayesian_authority(&artifact)?;
        compare_replay(&wire, &measured)?;
        Ok((artifact, measured))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_measured_protocol_requires_exact_original_fit_prior_and_sampler() {
        let fit = FitOptions::default();
        assert!(fit_scope(&fit).is_ok());
        assert!(fit_scope(&FitOptions { max_iterations: fit.max_iterations - 1, ..fit }).is_err());
        assert!(fit_scope(&FitOptions { tolerance: fit.tolerance * 2.0, ..fit }).is_err());
        assert!(
            fit_scope(&FitOptions { refuse_constraint_residual_above: Some(0.1), ..fit }).is_err()
        );
        let sampler = Options {
            chains: 4,
            warmup: 2048,
            draws: 4096,
            max_proposals: 5_000_000,
            seed: 817,
            credible_mass: 0.95,
        };
        assert!(bayesian_scope(&Prior::default(), &sampler).is_ok());
        assert!(bayesian_scope(&Prior { alpha: [2.0; 11], beta: [2.0; 11] }, &sampler).is_ok());
        let mut prior = Prior::default();
        prior.alpha[7] = 2.0;
        assert!(bayesian_scope(&prior, &sampler).is_err());
        for changed in [
            Options { chains: 3, ..sampler.clone() },
            Options { warmup: 2049, ..sampler.clone() },
            Options { draws: 4095, ..sampler.clone() },
            Options { max_proposals: 4_999_999, ..sampler.clone() },
            Options { credible_mass: 0.9, ..sampler },
        ] {
            assert!(bayesian_scope(&Prior::default(), &changed).is_err());
        }
    }
    #[test]
    fn measured_graph_label_bound_counts_utf8_bytes_and_refuses_before_fit() {
        assert!(label_scope(&["x".repeat(256), "é".repeat(128)]).is_ok());
        assert!(label_scope(&["x".repeat(257)]).is_err());
        assert!(label_scope(&["é".repeat(129)]).is_err());
        let mut graph = antecedent_estimate::nested_markov_binary::AdmgDeclaration::selected();
        graph.variables[0] = "x".repeat(257);
        let input = NestedMarkovInput { graph, regimes: vec![] };
        let error = NestedFisherArtifact::build_measured(
            &input,
            &FitOptions::default(),
            0.95,
            &ExecutionContext::for_tests(0),
        )
        .unwrap_err();
        assert!(
            matches!(error,IoError::Refused{code,..} if code==antecedent_core::reason_code!("cell_not_licensed"))
        );
        assert!(error.to_string().contains("256 UTF-8 byte bound"));
    }
    #[test]
    fn unsupported_nested_settings_refuse_before_even_invalid_data_is_fitted() {
        let input = NestedMarkovInput {
            graph: antecedent_estimate::nested_markov_binary::AdmgDeclaration::selected(),
            regimes: vec![],
        };
        let ctx = ExecutionContext::for_tests(0);
        let err = NestedFisherArtifact::build_measured(&input, &FitOptions::default(), 0.9, &ctx)
            .unwrap_err();
        assert!(err.to_string().contains("measured_scope"));
        let err = Artifact::build_measured(
            &input,
            &FitOptions::default(),
            Prior { alpha: [3.0; 11], beta: [3.0; 11] },
            Options {
                chains: 4,
                warmup: 2048,
                draws: 4096,
                max_proposals: 5_000_000,
                seed: 0,
                credible_mass: 0.95,
            },
            &ctx,
        )
        .unwrap_err();
        assert!(err.to_string().contains("measured_scope"));
    }
}

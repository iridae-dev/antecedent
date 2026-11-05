//! Measured whole-row recovery and whole-unit temporal constructions.
//! The original candidates retain their historical `not_measured` semantics.
use super::{
    MeasuredExpectation, MeasuredInference, ScalarExecution, authorize, compare_replay, decode,
};
use crate::IoError;
use crate::sampled_recovery_artifact::{
    SampledRecoveryArtifactWire, SampledRecoveryConsumeLimits, SampledRecoveryExpectation,
};
use crate::temporal_interval_artifact::{
    TemporalIntervalArtifactWire, TemporalIntervalConsumeLimits,
};
use antecedent_core::{ExecutionContext, IdentityDomain};

fn refusal(message: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("measured_interval.source_mismatch: {message}"),
    }
}

/// A basis for a construction no record measured: the estimator names it, so the calibration
/// slot reports `not_measured`. The interval itself is still reported.
fn tagged(
    measured: bool,
    mut basis: antecedent_core::CalibrationBasis,
) -> antecedent_core::CalibrationBasis {
    if !measured {
        basis.estimator = format!("{}+outside_measured_protocol", basis.estimator).into();
    }
    basis
}

impl SampledRecoveryArtifactWire {
    /// Independently verify the original recovery proof and whole-row `BCa` receipt,
    /// then bind its scalar effect to currently attesting evidence.
    /// # Errors
    /// Original proof/replay failures, a `not_measured` configuration, or absent/stale evidence.
    pub fn measure(
        bytes: &[u8],
        expected: &SampledRecoveryExpectation,
        limits: SampledRecoveryConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<MeasuredInference, IoError> {
        let replay = Self::consume_expecting(bytes, expected, limits, ctx)?;
        let weighted = replay.wire.catalog.bindings.iter().any(|binding| {
            binding.regime == replay.wire.query.observed_regime
                && binding.weights_snapshot.is_some()
        });
        let independent = replay.wire.catalog.bindings.iter().any(|binding| {
            binding.regime == replay.wire.query.observed_regime && binding.sampling == "independent"
        });
        let result = &replay.result;
        let config = &result.receipt.config;
        // Calibration annotates the interval: weighted rows, a non-independent sampling binding
        // or another configuration are reported and carry a `not_measured` slot.
        let measured = !weighted
            && independent
            && config.interval_method
                == antecedent_estimate::recovery_sampled::SampledIntervalMethod::Bca
            && config.replicates == 2000
            && result.failed_replicates == 0
            && config.max_failed_fraction.to_bits() == 0.0f64.to_bits()
            && config.normalization_tolerance.to_bits() == 0.1f64.to_bits()
            && config.small_cell_count.to_bits() == 5.0f64.to_bits()
            && config.treated_level.to_bits() == 1.0f64.to_bits()
            && config.control_level.to_bits() == 0.0f64.to_bits()
            && result.interval.level.to_bits() == 0.95f64.to_bits()
            && result.receipt.bca.is_some();
        let scope = serde_json::json!({
            "validation_design":"binary_confounded_missingness_scm",
            "evidence_interpretation":"finite validation grid for the checked binary missingness construction; arbitrary missingness graphs and sampling laws are not supported",
            "declared_model_assumptions":["independent observation rows","correct declared m-graph and recovery premises"],
            "sampling_assumptions_authenticated_from_rows":false,
            "rows":result.receipt.rows,"replicates":config.replicates,"method":"bootstrap_bca","level":result.interval.level,
            "failed_replicates":result.failed_replicates,"weighted_rows":weighted,"independent_rows":independent,"normalization_tolerance":config.normalization_tolerance,"small_cell_count":config.small_cell_count,
            "treated_level":config.treated_level,"control_level":config.control_level,
            "derivation_identity":result.receipt.derivation_identity,
            "reported_scalar":"recovered_effect"
        });
        authorize(
            "sampled_recovery",
            bytes.to_vec(),
            &replay.wire.premises_digest,
            &replay.wire.data_digest,
            scope,
            vec![ScalarExecution {
                name: "recovered_effect".into(),
                point: result.effect,
                interval: (result.interval.lower, result.interval.upper),
                basis: tagged(measured, result.calibration_basis()),
            }],
        )
    }
    /// Consume the measured envelope by rechecking its original recovery proof and full `BCa` work.
    /// # Errors
    /// Any source, expected identity, protocol, evidence or scalar mismatch.
    pub fn consume_measured(
        bytes: &[u8],
        expected: &MeasuredExpectation,
        limits: SampledRecoveryConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<MeasuredInference, IoError> {
        let wire = decode(bytes, expected)?;
        let measured = Self::measure(
            &wire.source,
            &SampledRecoveryExpectation {
                premises_digest: Some(expected.premises_digest.clone()),
                data_digest: Some(expected.data_digest.clone()),
            },
            limits,
            ctx,
        )?;
        compare_replay(&wire, &measured)?;
        Ok(measured)
    }
}

impl TemporalIntervalArtifactWire {
    /// Recompute the original balanced whole-unit studentized interval and resolve its actual basis.
    /// # Errors
    /// Source/replay failures, a `not_measured` target/configuration or absent/stale evidence.
    pub fn measure(
        bytes: &[u8],
        expected_seal: &str,
        limits: &TemporalIntervalConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<MeasuredInference, IoError> {
        let (wire, result) = Self::consume(bytes, limits, ctx)?;
        if wire.seal != expected_seal {
            return Err(refusal("original temporal source differs from retained identity"));
        }
        let config = &wire.config;
        let measured = wire.estimator.law.as_ref().is_some_and(|law| {
            wire.estimator.kind == "marginalized_initial_state"
                && wire.estimator.sequence == [0, 0]
                && config.method == "studentized"
                && config.replicates == 500
                && config.level.to_bits() == 0.95f64.to_bits()
                && config.min_units == 20
                && config.max_failed_fraction.to_bits() == 0.05f64.to_bits()
                && law.population == "target"
                && law.states.len() == 2
                && law.states[0].state == 0
                && law.states[1].state == 1
                && law.states[0].mass.to_bits() == 0.3f64.to_bits()
                && law.states[1].mass.to_bits() == 0.7f64.to_bits()
        }) && wire.panel.units.iter().all(|u| u.histories.len() == 16)
            && result.studentization.is_some();
        let premises = crate::digest_hex(IdentityDomain::Claim, &(&wire.estimator, &wire.config))?;
        let data = crate::digest_hex(IdentityDomain::Claim, &wire.panel)?;
        let scope = serde_json::json!({
            "validation_design":"independent_units_shared_shock_fixed_target",
            "evidence_interpretation":"finite validation grid for complete balanced binary histories; arbitrary serial histories and callbacks are not supported",
            "declared_model_assumptions":["independent units","whole-unit history ownership","correct native two-step response model"],
            "sampling_assumptions_authenticated_from_rows":false,
            "units":result.units,"sequence":wire.estimator.sequence,"measured_protocol":measured,
            "method":"studentized","level":result.level,"replicates":config.replicates,
            "min_units":config.min_units,"max_failed_fraction":config.max_failed_fraction,
            "reported_scalar":"response"
        });
        authorize(
            "temporal_interval",
            bytes.to_vec(),
            &premises,
            &data,
            scope,
            vec![ScalarExecution {
                name: "response".into(),
                point: result.point,
                interval: (result.lower, result.upper),
                basis: tagged(measured, result.calibration_basis()),
            }],
        )
    }
    /// Independently replay the original panel and rederive the measured envelope's scalar evidence.
    /// # Errors
    /// Any source/question/protocol/evidence/receipt mismatch.
    pub fn consume_measured(
        bytes: &[u8],
        expected: &MeasuredExpectation,
        limits: &TemporalIntervalConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<MeasuredInference, IoError> {
        let wire = decode(bytes, expected)?;
        let source = Self::decode(&wire.source)?;
        let measured = Self::measure(&wire.source, &source.seal, limits, ctx)?;
        compare_replay(&wire, &measured)?;
        Ok(measured)
    }
}

//! Native measured interval authority, separate from unchanged candidate artifacts.
//!
//! Serialized report flags and caller expectations confer no authority. Only
//! crate-owned producer/consumer adapters that derive each basis from a real
//! execution can construct [`MeasuredInference`]. Every reported scalar must
//! resolve to a current attesting non-boundary calibration record. Consumers
//! independently replay the original source and compare the complete report.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::IoError;
use crate::calibration::{CalibrationBasisWire, CalibrationKeyWire, CalibrationScopeWire};
use crate::contract_section::CalibrationSlotWire;
use antecedent_core::{CalibrationBasis, IdentityDomain};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Fixed outer format bound; caller limits can only narrow it.
pub const MAX_MEASURED_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
const MAX_SCALARS: usize = 32;
const FEATURE: &str = "measured_inference_v1";
pub(crate) mod nested;
pub(crate) mod recovery_temporal;
pub(crate) mod transport;

/// A caller's immutable expected source and reported question. This is a
/// declaration, never a license or a substitute for actual source replay.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredExpectation {
    /// Executed adapter route.
    pub route: String,
    /// Canonical original candidate bytes digest.
    pub candidate_digest: String,
    /// Original native premise identity.
    pub premises_digest: String,
    /// Original native data identity.
    pub data_digest: String,
    /// Exact requested nominal level.
    pub level: f64,
    /// Ordered names of every reported scalar.
    pub scalars: Vec<String>,
    /// Whole measured report identity.
    pub seal: String,
}
/// One actual execution's scalar and governing record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredScalarReport {
    /// Exact scalar functional name.
    pub name: String,
    /// Actual point estimate.
    pub point: f64,
    /// Actual interval lower endpoint.
    pub lower: f64,
    /// Actual interval upper endpoint.
    pub upper: f64,
    /// Actual nominal level.
    pub level: f64,
    /// Actual resolver output, including native basis and measured Git commit.
    pub calibration: CalibrationSlotWire,
}
/// Readable report; deserializing it does not construct native authority.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredInferenceReport {
    /// Format version.
    pub version: u32,
    /// Native route.
    pub route: String,
    /// `calibrated` only after every actual basis resolves to attesting evidence.
    pub calibration: String,
    /// Complete reported scalar list, with individual record bindings.
    pub scalars: Vec<MeasuredScalarReport>,
    /// Adapter-derived checked facts and explicitly declared model assumptions.
    pub validated_scope: serde_json::Value,
    /// Complete caller-retainable expected question/source identity.
    pub identity: MeasuredExpectation,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Untrusted readable envelope DTO. Decoding or constructing it never grants authority.
pub struct MeasuredEnvelopeWire {
    /// Format version.
    pub version: u32,
    /// Required wire semantics.
    pub required_features: Vec<String>,
    /// Original candidate source bytes.
    pub source: Vec<u8>,
    /// Readable report, not an authorized native result.
    pub report: MeasuredInferenceReport,
}

#[derive(Serialize)]
struct EnvelopeView<'a> {
    version: u32,
    required_features: [&'static str; 1],
    source: &'a [u8],
    report: &'a MeasuredInferenceReport,
}

/// Factory-only native authorization. It has no public constructor or
/// deserializer, and holds immutable original source and verified report bytes.
#[derive(Clone, Debug)]
pub struct MeasuredInference {
    report: MeasuredInferenceReport,
    bytes: Arc<[u8]>,
    source: Arc<[u8]>,
}
impl MeasuredInference {
    /// Actual verified report.
    #[must_use]
    pub const fn report(&self) -> &MeasuredInferenceReport {
        &self.report
    }
    /// Original unmodified candidate artifact (including its candidate standing).
    #[must_use]
    pub fn source_artifact(&self) -> &[u8] {
        &self.source
    }
    /// Immutable canonical measured envelope; exporting never refits.
    #[must_use]
    pub fn export(&self) -> &[u8] {
        &self.bytes
    }
    /// Source/question expectation for independent fresh consumption.
    #[must_use]
    pub const fn expectation(&self) -> &MeasuredExpectation {
        &self.report.identity
    }

    /// Independently replay a supported original source and re-resolve every scalar.
    /// Caller expectations constrain identity; they never supply calibration authority.
    /// # Errors
    /// Unsupported route, changed source/question/result, original native refusal,
    /// exceeded native bounds, or evidence that is not currently attesting.
    pub fn consume(
        bytes: &[u8],
        expected: &MeasuredExpectation,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, IoError> {
        match expected.route.as_str() {
            "joint_bayesian" => crate::joint_bayesian_transport_artifact::JointBayesianArtifactWire::consume_measured(bytes, expected, crate::joint_bayesian_transport_artifact::JointBayesianConsumeLimits::default(), ctx).map(|(_, _, measured)| measured),
            "learned_joint" => crate::learned_joint_transport_artifact::LearnedJointArtifactWire::consume_measured(bytes, expected, crate::learned_joint_transport_artifact::LearnedJointConsumeLimits::default(), ctx).map(|(_, _, measured)| measured),
            "nested_fisher" => crate::nested_markov_artifact::NestedFisherArtifact::consume_measured(bytes, expected, crate::nested_markov_artifact::NestedMarkovConsumeLimits::default(), ctx).map(|(_, measured)| measured),
            "nested_bayesian" => crate::nested_markov_bayesian_artifact::Artifact::consume_measured(bytes, expected, crate::nested_markov_bayesian_artifact::Limits::default(), ctx).map(|(_, measured)| measured),
            "sampled_recovery" => crate::sampled_recovery_artifact::SampledRecoveryArtifactWire::consume_measured(bytes, expected, crate::sampled_recovery_artifact::SampledRecoveryConsumeLimits::default(), ctx),
            "temporal_interval" => crate::temporal_interval_artifact::TemporalIntervalArtifactWire::consume_measured(bytes, expected, &crate::temporal_interval_artifact::TemporalIntervalConsumeLimits::default(), ctx),
            _ => Err(refusal("measured_inference.unsupported_route", "no measured native consumer is registered for the requested route")),
        }
    }
}

/// Scalar declaration for stateless record lookup/report construction. A caller's
/// basis can construct readable metadata only, never opaque native authority.
pub struct ScalarExecution {
    /// Reported scalar name.
    pub name: String,
    /// Declared point estimate.
    pub point: f64,
    /// Declared interval endpoints.
    pub interval: (f64, f64),
    /// Declared calibration basis; a route must derive it independently before authorization.
    pub basis: CalibrationBasis,
}
fn refusal(detail: &str, message: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("cell_not_licensed"),
        message: format!("{detail}: {message}"),
    }
}
fn basis_wire(b: &CalibrationBasis) -> CalibrationBasisWire {
    CalibrationBasisWire {
        key: CalibrationKeyWire {
            query: b.query.to_string(),
            graph_class: b.graph_class.to_string(),
            structure: b.structure.to_string(),
            modality: b.modality.to_string(),
            inference: b.inference.to_string(),
            estimator: b.estimator.to_string(),
            interval_method: b.interval_method.to_string(),
            se_kind: b.se_kind.to_string(),
            dependence: b.dependence.to_string(),
            posterior: b.posterior.to_string(),
            functional: b.functional.to_string(),
            level: b.level,
            identification: b.identification.to_string(),
        },
        scope: CalibrationScopeWire {
            row_count: b.row_count,
            replicates_ok: b.replicates_ok,
            posterior_draws: b.posterior_draws,
            unidentified_mass: b.unidentified_mass,
        },
    }
}
/// Canonical readable report seal. A checksum does not grant inference authority.
/// # Errors
/// Canonical report serialization failure.
pub fn report_seal(report: &MeasuredInferenceReport) -> Result<String, IoError> {
    let mut identity = report.identity.clone();
    identity.seal.clear();
    Ok(crate::identity::digest_wire(
        IdentityDomain::Claim,
        &(
            FEATURE,
            report.version,
            &report.route,
            &report.calibration,
            &report.scalars,
            &report.validated_scope,
            identity,
        ),
    )?
    .to_hex())
}

/// Build readable record-bound metadata from supplied declarations. This is
/// stateless lookup and formatting only: arbitrary caller bases never construct
/// [`MeasuredInference`] or authenticate an execution/model/sampling design.
/// # Errors
/// Invalid/bounded scalar inputs or a basis without current attesting evidence.
pub fn build_report(
    route: &str,
    source: &[u8],
    premises: &str,
    data: &str,
    scope: serde_json::Value,
    executions: Vec<ScalarExecution>,
) -> Result<MeasuredInferenceReport, IoError> {
    if executions.is_empty()
        || executions.len() > MAX_SCALARS
        || source.len() > MAX_MEASURED_ARTIFACT_BYTES
    {
        return Err(refusal(
            "measured_inference.bounds_exceeded",
            "source/scalar count exceeds native measured bounds",
        ));
    }
    let level = executions[0].basis.level;
    let mut names = std::collections::BTreeSet::new();
    let mut scalars = Vec::with_capacity(executions.len());
    for execution in executions {
        if !names.insert(execution.name.clone())
            || !execution.point.is_finite()
            || !execution.interval.0.is_finite()
            || !execution.interval.1.is_finite()
            || execution.interval.0 > execution.interval.1
            || execution.basis.level.to_bits() != level.to_bits()
        {
            return Err(refusal(
                "measured_inference.invalid_report",
                "actual reported scalar is invalid or has a different nominal level",
            ));
        }
        let basis = basis_wire(&execution.basis);
        let slot = crate::calibration::calibration_slot(&basis);
        if slot.status != "calibrated" || slot.record_id.is_none() || slot.calibration_sha.is_none()
        {
            return Err(refusal(
                "measured_inference.record_not_attesting",
                &format!(
                    "{}: {}",
                    execution.name,
                    slot.reason.as_deref().unwrap_or("no attesting record")
                ),
            ));
        }
        scalars.push(MeasuredScalarReport {
            name: execution.name,
            point: execution.point,
            lower: execution.interval.0,
            upper: execution.interval.1,
            level,
            calibration: slot,
        });
    }
    let identity = MeasuredExpectation {
        route: route.into(),
        candidate_digest: crate::identity::digest_canonical(IdentityDomain::Claim, source).to_hex(),
        premises_digest: premises.into(),
        data_digest: data.into(),
        level,
        scalars: scalars.iter().map(|s| s.name.clone()).collect(),
        seal: String::new(),
    };
    let mut report = MeasuredInferenceReport {
        version: 1,
        route: route.into(),
        calibration: "calibrated".into(),
        scalars,
        validated_scope: scope,
        identity,
    };
    report.identity.seal = report_seal(&report)?;
    Ok(report)
}

/// Encode readable DTOs without constructing any native inference authority.
/// # Errors
/// Invalid identities, canonical serialization failure or the fixed byte bound.
pub fn encode_report(source: &[u8], report: &MeasuredInferenceReport) -> Result<Vec<u8>, IoError> {
    if source.len() > MAX_MEASURED_ARTIFACT_BYTES
        || report.identity.seal != report_seal(report)?
        || report.identity.candidate_digest
            != crate::identity::digest_canonical(IdentityDomain::Claim, source).to_hex()
    {
        return Err(refusal(
            "measured_inference.artifact_mismatch",
            "readable source/report identity is invalid",
        ));
    }
    let bytes =
        crate::to_cbor(&EnvelopeView { version: 1, required_features: [FEATURE], source, report })?;
    if bytes.len() > MAX_MEASURED_ARTIFACT_BYTES {
        return Err(refusal(
            "measured_inference.bounds_exceeded",
            "measured envelope exceeds 64 MiB",
        ));
    }
    Ok(bytes)
}

/// Crate-owned adapters derive these inputs from actual successful fit/replay.
/// Public report utilities cannot call this opaque authority factory.
pub(crate) fn authorize(
    route: &str,
    source: Vec<u8>,
    premises: &str,
    data: &str,
    scope: serde_json::Value,
    executions: Vec<ScalarExecution>,
) -> Result<MeasuredInference, IoError> {
    let report = build_report(route, &source, premises, data, scope, executions)?;
    let bytes = encode_report(&source, &report)?;
    Ok(MeasuredInference { report, source: Arc::from(source), bytes: Arc::from(bytes) })
}

/// Decode an untrusted envelope only. This never grants measured authority.
/// # Errors
/// Byte bound, canonical wire, checksum or caller expected-identity mismatch.
pub fn decode(
    bytes: &[u8],
    expected: &MeasuredExpectation,
) -> Result<MeasuredEnvelopeWire, IoError> {
    if bytes.len() > MAX_MEASURED_ARTIFACT_BYTES {
        return Err(refusal(
            "measured_inference.bounds_exceeded",
            "measured envelope exceeds 64 MiB",
        ));
    }
    let wire: MeasuredEnvelopeWire = crate::from_cbor(bytes)?;
    if wire.version != 1
        || wire.required_features != [FEATURE]
        || wire.report.version != 1
        || wire.report.identity != *expected
        || wire.report.route != expected.route
        || wire.report.identity.seal != report_seal(&wire.report)?
        || wire.report.identity.candidate_digest
            != crate::identity::digest_canonical(IdentityDomain::Claim, &wire.source).to_hex()
        || wire.report.scalars.len() > MAX_SCALARS
        || crate::to_cbor(&wire)? != bytes
    {
        return Err(refusal(
            "measured_inference.artifact_mismatch",
            "measured envelope differs from expected identity or canonical source",
        ));
    }
    Ok(wire)
}
/// Compare to fresh native authorization after the original independent replay.
pub(crate) fn compare_replay(
    wire: &MeasuredEnvelopeWire,
    measured: &MeasuredInference,
) -> Result<(), IoError> {
    if crate::to_cbor(&wire.report)? != crate::to_cbor(measured.report())?
        || wire.source != measured.source_artifact()
    {
        return Err(refusal(
            "measured_inference.artifact_mismatch",
            "actual native source, basis, scalar or governing record differs on replay",
        ));
    }
    Ok(())
}

/// The measured posterior-quantile construction: equal-tailed type-7 quantiles.
pub(crate) fn posterior_interval(mut draws: Vec<f64>, level: f64) -> Result<(f64, f64), IoError> {
    if draws.len() < 2 || draws.iter().any(|x| !x.is_finite()) || !(0.0 < level && level < 1.0) {
        return Err(refusal(
            "measured_inference.invalid_report",
            "actual posterior interval inputs are invalid",
        ));
    }
    draws.sort_by(f64::total_cmp);
    let quantile = |p: f64| {
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "bounded native posterior draw indexing"
        )]
        let h = p * (draws.len() - 1) as f64;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "bounded native posterior draw indexing"
        )]
        let lo = h.floor() as usize;
        #[allow(clippy::cast_precision_loss, reason = "bounded native posterior draw indexing")]
        let fraction = h - lo as f64;
        draws[lo] + fraction * (draws[(lo + 1).min(draws.len() - 1)] - draws[lo])
    };
    Ok((quantile((1.0 - level) / 2.0), quantile(1.0 - (1.0 - level) / 2.0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn posterior_interval_matches_independent_linear_quantile_oracle() {
        let interval = posterior_interval(vec![30.0, 0.0, 10.0, 20.0], 0.5).unwrap();
        assert!((interval.0 - 7.5).abs() < f64::EPSILON);
        assert!((interval.1 - 22.5).abs() < f64::EPSILON);
        assert!(posterior_interval(vec![f64::NAN, 1.0], 0.95).is_err());
        assert!(posterior_interval(vec![1.0], 0.95).is_err());
        assert!(posterior_interval(vec![1.0, 2.0], 1.0).is_err());
    }
    #[test]
    fn sealed_calibrated_flags_and_forged_record_never_grant_source_authority() {
        let source = vec![1, 2, 3];
        for route in ["joint_bayesian", "learned_joint", "nested_fisher", "nested_bayesian"] {
            let mut slot = CalibrationSlotWire::unavailable("invented");
            slot.status = "calibrated".into();
            slot.record_id = Some("invented.record".into());
            slot.calibration_sha = Some("0".repeat(40));
            let mut report = MeasuredInferenceReport {
                version: 1,
                route: route.into(),
                calibration: "calibrated".into(),
                scalars: vec![MeasuredScalarReport {
                    name: "target_effect".into(),
                    point: 2.0,
                    lower: 1.0,
                    upper: 3.0,
                    level: 0.95,
                    calibration: slot,
                }],
                validated_scope: serde_json::json!({"forged":true}),
                identity: MeasuredExpectation {
                    route: route.into(),
                    candidate_digest: crate::identity::digest_canonical(
                        IdentityDomain::Claim,
                        &source,
                    )
                    .to_hex(),
                    premises_digest: "a".repeat(64),
                    data_digest: "b".repeat(64),
                    level: 0.95,
                    scalars: vec!["target_effect".into()],
                    seal: String::new(),
                },
            };
            report.identity.seal = report_seal(&report).unwrap();
            let expected = report.identity.clone();
            let bytes = crate::to_cbor(&MeasuredEnvelopeWire {
                version: 1,
                required_features: vec![FEATURE.into()],
                source: source.clone(),
                report,
            })
            .unwrap();
            // A valid checksum is only an untrusted declaration, not a native factory.
            assert!(decode(&bytes, &expected).is_ok());
            let ctx = antecedent_core::ExecutionContext::for_tests(1);
            assert!(MeasuredInference::consume(&bytes, &expected, &ctx).is_err());
            let mut changed = expected;
            changed.level = 0.9;
            assert!(decode(&bytes, &changed).is_err());
        }
    }
}

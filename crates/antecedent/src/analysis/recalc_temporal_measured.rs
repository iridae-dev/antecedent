//! Measured whole-unit inference from the ORIGINAL checked temporal source.
//! The native source proof and both effect arms are replayed; plain-panel
//! calibration is never substituted for checked-source evidence.
use super::recalc_temporal::{TemporalRunError, TemporalSession, consume_temporal_recalc_artifact};
use antecedent_core::{ExecutionContext, RngFactory, reason_code};
use antecedent_estimate::temporal_dependent_interval::{DependentIntervalConfig, IntervalMethod};
use antecedent_io::IoError;
use antecedent_io::measured_inference::{
    MeasuredExpectation, MeasuredInferenceReport, ScalarExecution, build_report, decode,
    encode_report,
};
use antecedent_io::temporal_interval_artifact::{IntervalConfigWire, IntervalResultWire};
use antecedent_io::temporal_recalc_artifact::{TemporalFunctionalWire, TemporalRecalcArtifactWire};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
const MAX_SOURCE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckedSource {
    version: u32,
    required_features: Vec<String>,
    calibration: String,
    source: Vec<u8>,
    config: IntervalConfigWire,
    result: IntervalResultWire,
}
/// Native factory-only measured original-source interval. Serialized report
/// flags or caller-provided bases cannot construct this type.
#[derive(Clone, Debug)]
pub struct CheckedMeasuredTemporal {
    report: MeasuredInferenceReport,
    bytes: Arc<[u8]>,
    source: Arc<[u8]>,
}
fn refused(message: &str) -> TemporalRunError {
    IoError::Refused {
        code: reason_code!("cell_not_licensed"),
        message: format!("checked_temporal.protocol_not_measured: {message}"),
    }
    .into()
}
// Every method uses the same complete balanced history design as its calibration.
// Counting sixteen rows alone would admit duplicated or missing cells in basic/percentile.
fn complete_binary_unit(unit: &antecedent_io::temporal_recalc_artifact::TemporalUnitWire) -> bool {
    if unit.histories.len() != 16 {
        return false;
    }
    let mut seen = 0u16;
    for h in &unit.histories {
        if [h.s0, h.a1, h.l2, h.a2].iter().any(|state| *state > 1)
            || ![0.0f64.to_bits(), (-0.0f64).to_bits(), 1.0f64.to_bits()].contains(&h.y.to_bits())
        {
            return false;
        }
        let bit = 1u16 << (h.s0 * 8 + h.a1 * 4 + h.l2 * 2 + h.a2);
        if seen & bit != 0 {
            return false;
        }
        seen |= bit;
    }
    seen == u16::MAX
}
fn preflight(
    source: &TemporalRecalcArtifactWire,
    config: &DependentIntervalConfig,
    ctx: &ExecutionContext,
    source_bytes: usize,
) -> Result<&'static str, TemporalRunError> {
    let request = &source.request;
    let effect = matches!(
        &request.functional,
        TemporalFunctionalWire::Effect { active: [1, 1], control: [0, 0] }
    );
    let response =
        matches!(&request.functional, TemporalFunctionalWire::Response { sequence: [0, 0] });
    if !(effect || response)
        || (response && config.method != IntervalMethod::Studentized)
        || config.level.to_bits() != 0.95f64.to_bits()
        || config.replicates != 500
        || config.min_units != 20
        || config.max_failed_fraction.to_bits() != 0.05f64.to_bits()
        || request.initial_state.map(f64::to_bits) != [0.3f64.to_bits(), 0.7f64.to_bits()]
        || request.horizon != 2
        || request.lag_alignment != [0, 1, 2, 2, 2]
        || !request.bidirected.is_empty()
        || request.selection_targets != [0]
        || request.edges.len() != 10
        || [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
            .iter()
            .any(|edge| !request.edges.contains(edge))
        || !(64..=256).contains(&request.units.len())
        || request.units.iter().any(|u| !complete_binary_unit(u))
    {
        return Err(refused(
            "only the measured complete binary-history 95% response (studentized) or paired effect (studentized, percentile, basic), 500 draws and target law (0.3, 0.7), is licensed",
        ));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(IoError::Refused {
            code: reason_code!("cancelled_no_claim"),
            message: "checked temporal interval cancelled before replay".into(),
        }
        .into());
    }
    let rows = request.units.iter().map(|u| u.histories.len()).sum::<usize>();
    let planned = u64::try_from(rows)
        .unwrap_or(u64::MAX)
        .saturating_mul(2048)
        .saturating_add(u64::try_from(source_bytes).unwrap_or(u64::MAX).saturating_mul(8))
        .saturating_add(4 * 1024 * 1024);
    if ctx.memory.hard_limit_bytes.is_some_and(|limit| limit < planned) {
        return Err(refused("planned source replay and interval receipt exceed memory budget"));
    }
    Ok(if effect { "effect" } else { "response" })
}
impl CheckedMeasuredTemporal {
    /// Produce from a retained checked session and its ORIGINAL source proof.
    /// # Errors
    /// Unmeasured protocol, source/interval refusal, cancellation, limits or stale evidence.
    pub fn produce(
        session: &TemporalSession,
        config: &DependentIntervalConfig,
        ctx: &ExecutionContext,
    ) -> Result<Self, TemporalRunError> {
        let original = session.export_result()?;
        if original.len() > MAX_SOURCE_BYTES {
            return Err(refused("original source exceeds 16 MiB"));
        }
        let source = TemporalRecalcArtifactWire::decode(&original)?;
        let name = preflight(&source, config, ctx, original.len())?;
        let mut producing = ctx.clone();
        producing.rng = RngFactory::from_seed(source.seed);
        let result = session.candidate_interval_replay(config, &producing)?;
        let bundle = CheckedSource {
            version: 1,
            required_features: vec!["checked_temporal_measured_source_v1".into()],
            calibration: "unmeasured".into(),
            source: original,
            config: IntervalConfigWire::from_config(config),
            result: IntervalResultWire::from_interval(&result),
        };
        let bytes = antecedent_io::to_cbor(&bundle)?;
        let scope = serde_json::json!({
            "validation_design":"independent_binary_units_checked_initial_shift",
            "evidence_interpretation":"finite validation grid for the original checked binary two-step source; no arbitrary temporal-model or callback coverage",
            "declared_model_assumptions":["independent units","correct source graph and initial-state-only selection","complete binary two-step histories"],
            "sampling_assumptions_authenticated_from_rows":false,
            "functional":source.request.functional,"target_law":source.request.initial_state,
            "source_data_digest":source.data_digest,"source_premises_digest":source.premises_digest,
            "snapshot_id":source.request.snapshot_id,"initial_state_id":source.request.initial_state_id,
            "producing_seed":source.seed,"bootstrap_seed":config.seed,"config":bundle.config,
            "units":result.units,"histories_per_unit":16,"reported_scalar":name
        });
        let report = build_report(
            "checked_temporal",
            &bytes,
            &source.premises_digest,
            &source.data_digest,
            scope,
            vec![ScalarExecution {
                name: name.into(),
                point: result.point,
                interval: (result.lower, result.upper),
                basis: result.calibration_basis(),
            }],
        )?;
        let envelope = encode_report(&bytes, &report)?;
        Ok(Self { report, bytes: Arc::from(envelope), source: Arc::from(bytes) })
    }
    /// Consume by freshly reidentifying/refitting the original source and both arms,
    /// then replaying every whole-unit draw and rederiving all scalar evidence.
    /// # Errors
    /// Any changed source, full interval receipt, expected identity, scope or evidence.
    pub fn consume(
        bytes: &[u8],
        expected: &MeasuredExpectation,
        ctx: &ExecutionContext,
    ) -> Result<Self, TemporalRunError> {
        let wire = decode(bytes, expected)?;
        let bundle: CheckedSource = antecedent_io::from_cbor(&wire.source)?;
        if bundle.version != 1
            || bundle.required_features != ["checked_temporal_measured_source_v1"]
            || bundle.calibration != "unmeasured"
            || bundle.source.len() > MAX_SOURCE_BYTES
            || antecedent_io::to_cbor(&bundle)? != wire.source
        {
            return Err(refused("foreign or noncanonical checked source bundle"));
        }
        let original = TemporalRecalcArtifactWire::decode(&bundle.source)?;
        let config = bundle.config.to_config()?;
        preflight(&original, &config, ctx, bundle.source.len())?;
        let mut producing = ctx.clone();
        producing.rng = RngFactory::from_seed(original.seed);
        let (session, _) = consume_temporal_recalc_artifact(&bundle.source, &producing)?;
        let measured = Self::produce(&session, &config, &producing)?;
        if measured.export() != bytes {
            return Err(refused(
                "original proof, complete interval receipt or calibrated scalar differs on independent replay",
            ));
        }
        Ok(measured)
    }
    /// Actual verified scalar report.
    #[must_use]
    pub const fn report(&self) -> &MeasuredInferenceReport {
        &self.report
    }
    /// Original source/proof/config/complete interval bundle; standing remains unmeasured.
    #[must_use]
    pub fn source_artifact(&self) -> &[u8] {
        &self.source
    }
    /// Immutable canonical measured envelope; export performs no fits.
    #[must_use]
    pub fn export(&self) -> &[u8] {
        &self.bytes
    }
    /// Retain this expectation separately for fresh independent consumption.
    #[must_use]
    pub const fn expectation(&self) -> &MeasuredExpectation {
        &self.report.identity
    }
    /// Readable original checked source/proof and complete interval receipt.
    /// # Errors
    /// Internal stored encoding failure.
    pub fn source_report(&self) -> Result<serde_json::Value, IoError> {
        let bundle: CheckedSource = antecedent_io::from_cbor(&self.source)?;
        let source = TemporalRecalcArtifactWire::decode(&bundle.source)?;
        Ok(
            serde_json::json!({"calibration":"unmeasured","source":source,"config":bundle.config,"result":bundle.result}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::recalc_temporal::{
        TemporalFunctional, TemporalHistoryWire, TemporalRequest, TemporalUnitWire,
        execute_temporal_with_receipt,
    };
    use super::*;
    use antecedent_core::IdentityDomain;
    use antecedent_io::identity::digest_canonical;
    use antecedent_io::measured_inference::report_seal;

    fn session() -> TemporalSession {
        let mut state = 731u64;
        let mut units = Vec::new();
        for unit_id in 0..80u64 {
            let mut histories = Vec::new();
            for s0 in 0..2u32 {
                for a1 in 0..2u32 {
                    for l2 in 0..2u32 {
                        for a2 in 0..2u32 {
                            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                            let y = f64::from(u32::from(
                                (state >> 32) % 10 < u64::from(3 + s0 + l2 + a1 + a2),
                            ));
                            histories.push(TemporalHistoryWire {
                                time_id: u64::try_from(histories.len()).unwrap() * 3,
                                s0,
                                a1,
                                l2,
                                a2,
                                y,
                            });
                        }
                    }
                }
            }
            units.push(TemporalUnitWire { unit_id, histories });
        }
        let request = TemporalRequest {
            edges: vec![
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 3),
                (2, 4),
                (3, 4),
            ],
            bidirected: vec![],
            selection_targets: vec![0],
            horizon: 2,
            lag_alignment: [0, 1, 2, 2, 2],
            period: (0, 49),
            units,
            snapshot_id: "measured-artifact-oracle".into(),
            initial_state: [0.3, 0.7],
            initial_state_id: "fixed-target-law".into(),
            functional: TemporalFunctional::Effect { active: [1, 1], control: [0, 0] },
            benefit_per_unit: 1.0,
            cost: 0.0,
        };
        let mut session = TemporalSession::new();
        execute_temporal_with_receipt(&mut session, &request, &ExecutionContext::for_tests(731))
            .unwrap();
        session
    }

    #[test]
    fn measured_checked_artifact_replays_and_rejects_resealed_scalar_and_full_receipt() {
        let context = ExecutionContext::for_tests(99_991);
        let config = DependentIntervalConfig {
            method: IntervalMethod::Studentized,
            seed: 901,
            ..DependentIntervalConfig::default()
        };
        let measured = CheckedMeasuredTemporal::produce(&session(), &config, &context).unwrap();
        let replayed =
            CheckedMeasuredTemporal::consume(measured.export(), measured.expectation(), &context)
                .unwrap();
        assert_eq!(replayed.export(), measured.export());
        assert_eq!(replayed.source_artifact(), measured.source_artifact());
        assert_eq!(measured.report().scalars[0].calibration.status, "calibrated");

        // Legitimate checksum/expectation updates must not license altered scalar values.
        let mut report = measured.report().clone();
        report.scalars[0].point += 0.125;
        report.identity.seal = report_seal(&report).unwrap();
        let forged = encode_report(measured.source_artifact(), &report).unwrap();
        assert!(decode(&forged, &report.identity).is_ok());
        assert!(CheckedMeasuredTemporal::consume(&forged, &report.identity, &context).is_err());

        // Change an internal per-draw pivot, leaving point/endpoints superficially intact.
        // Bind the altered source bytes and recompute the outer seal; only full replay rejects it.
        let mut source: CheckedSource =
            antecedent_io::from_cbor(measured.source_artifact()).unwrap();
        let pivot = &mut source.result.studentization.as_mut().unwrap().pivots[0];
        *pivot = Some(pivot.unwrap() + 0.125);
        let changed_source = antecedent_io::to_cbor(&source).unwrap();
        let mut report = measured.report().clone();
        report.identity.candidate_digest =
            digest_canonical(IdentityDomain::Claim, &changed_source).to_hex();
        report.identity.seal = report_seal(&report).unwrap();
        let forged = encode_report(&changed_source, &report).unwrap();
        assert!(decode(&forged, &report.identity).is_ok());
        assert!(CheckedMeasuredTemporal::consume(&forged, &report.identity, &context).is_err());
    }
}

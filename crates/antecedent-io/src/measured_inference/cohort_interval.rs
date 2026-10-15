//! Measured interval of the declared-cohort mean of a natively fitted linear effect.
//!
//! The source is a pair of independently replayable artifacts: the compact export of the
//! native fit (outcome on treatment-by-basis, HC3 sandwich covariance, retained `n` and
//! `p` in its identity) and the cohort decision source built from exactly that export.
//! The consumer re-verifies both under their retained identities, recomputes the
//! weighted cohort mean and its model-based standard error from the stored rows and
//! coefficients, and issues `point +/- t_{n-p} se`. The reported scalar must resolve to
//! a currently attesting calibration record of the native construction; a declared
//! (caller-built) export, a shifted cohort and an unmeasured level are refused.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    MeasuredExpectation, MeasuredInference, ScalarExecution, authorize, compare_replay, decode,
};
use crate::IoError;
use antecedent_core::{ExecutionContext, IdentityDomain};
use antecedent_estimate::cohort_linear_effect::{
    CohortIntervalRefusal, FUNCTIONAL_ID, INTERVAL_METHOD, INTERVAL_SCOPE, MAX_FIT_FEATURES,
    MAX_FIT_ROWS, NativeCohortFit, NativeFitRows, calibration_basis, native_cohort_interval,
};
use antecedent_io_core::cohort_decision::{CohortDecisionSource, CohortExpectation};
use antecedent_io_core::compact_export::{CompactExport, ExportLimits};
use serde::{Deserialize, Serialize};

/// Route name the measured envelope is consumed under.
pub const ROUTE: &str = "cohort_interval";

fn refused(detail: &str, message: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("cell_not_licensed"),
        message: format!("{detail}: {message}"),
    }
}

fn from_interval(error: &CohortIntervalRefusal) -> IoError {
    IoError::Refused {
        code: error.code,
        message: format!("{}: {} ({})", error.detail, error.remedy, error.subject),
    }
}

/// The rows a native fit was computed from, retained so a consumer replays the fit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitRowsWire {
    /// Outcome, one per row.
    pub outcome: Vec<f64>,
    /// Treatment, one per row.
    pub treatment: Vec<f64>,
    /// Modifier columns aligned with `feature_ids`.
    pub features: Vec<Vec<f64>>,
    /// Modifier variable ids.
    pub feature_ids: Vec<String>,
}

/// The retained source of a measured cohort interval: both artifacts, the fit rows and
/// the identities a consumer retained out of band.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CohortIntervalArtifactWire {
    /// Format version.
    pub version: u32,
    /// Compact export bytes of the native fit.
    pub export: Vec<u8>,
    /// Identity of the export.
    pub export_identity: String,
    /// The rows the native fit was computed from; the consumer refits and requires the
    /// export, its marker, coefficients and covariance to reproduce exactly.
    pub fit_rows: FitRowsWire,
    /// Cohort decision source bytes.
    pub source: Vec<u8>,
    /// Identity of the decision source.
    pub source_identity: String,
    /// Requested nominal two-sided level.
    pub level: f64,
}

impl CohortIntervalArtifactWire {
    /// Serialize the source of a measured interval.
    ///
    /// # Errors
    /// An export or source that cannot be serialized.
    pub fn encode(
        native: &NativeCohortFit,
        source: &CohortDecisionSource,
        level: f64,
    ) -> Result<Vec<u8>, IoError> {
        let export = native.export();
        let rows = native.rows();
        let export_bytes = export
            .to_bytes("cohort_interval_model")
            .map_err(|e| refused("cohort_decision.fitted_model_invalid", &e.to_string()))?;
        let source_bytes = source
            .to_bytes("cohort_interval_source")
            .map_err(|e| refused("cohort_decision.source_invalid", &e.to_string()))?;
        crate::to_cbor(&Self {
            version: 1,
            export: export_bytes,
            export_identity: export.identity().to_owned(),
            fit_rows: FitRowsWire {
                outcome: rows.outcome.clone(),
                treatment: rows.treatment.clone(),
                features: rows.features.clone(),
                feature_ids: rows.feature_ids.clone(),
            },
            source: source_bytes,
            source_identity: source.identity().to_owned(),
            level,
        })
    }

    /// Replay both artifacts and bind the interval to the current calibration evidence.
    ///
    /// # Errors
    /// A changed or tampered artifact, a declared export, a shifted cohort, an invalid
    /// level, or a construction with no currently attesting record.
    pub fn measure(bytes: &[u8], ctx: &ExecutionContext) -> Result<MeasuredInference, IoError> {
        let wire: Self = crate::from_cbor(bytes)?;
        if wire.version != 1 || crate::to_cbor(&wire)? != bytes {
            return Err(refused(
                "measured_inference.artifact_mismatch",
                "cohort interval source is not canonical version 1",
            ));
        }
        let export =
            CompactExport::consume(&wire.export, &ExportLimits::default(), &wire.export_identity)
                .map_err(|e| refused("cohort_decision.fitted_model_invalid", &e.to_string()))?;
        if wire.fit_rows.outcome.len() > MAX_FIT_ROWS
            || wire.fit_rows.features.len() > MAX_FIT_FEATURES
        {
            return Err(refused("measured_inference.bounds_exceeded", "fit rows exceed bounds"));
        }
        // The native-fit marker is never trusted: refit from the retained rows and
        // require the stored export to be exactly what the fit produces.
        let native = NativeCohortFit::replay(
            NativeFitRows {
                outcome: wire.fit_rows.outcome.clone(),
                treatment: wire.fit_rows.treatment.clone(),
                features: wire.fit_rows.features.clone(),
                feature_ids: wire.fit_rows.feature_ids.clone(),
            },
            &export,
        )
        .map_err(|_| {
            refused(
                "cohort_decision.interval_requires_native_fit",
                "the retained rows do not reproduce the export; a declared export is point-only",
            )
        })?;
        let source = CohortDecisionSource::consume_with(
            &wire.source,
            &CohortExpectation::for_identity(wire.source_identity.clone()),
            ctx,
        )
        .map_err(|e| refused(e.detail, &e.to_string()))?;
        let interval =
            native_cohort_interval(&native, &source, wire.level).map_err(|e| from_interval(&e))?;
        let body = export.body();
        let scope = serde_json::json!({
            "validation_design": "linear_conditional_effect_ols_hc3_randomized_and_observed_confounded",
            "evidence_interpretation": "finite validation grid for a natively fitted linear effect basis of two to four modifiers and declared cohorts of 50 to 200 rows; misspecified effect bases and high-leverage cohorts are named boundaries",
            "declared_model_assumptions": [
                "treatment randomized or ignorable given exactly the declared modifiers",
                "outcome mean linear in the modifiers under each arm and a linear effect basis",
                "independent rows"
            ],
            "interval_scope": INTERVAL_SCOPE,
            "interval_method": INTERVAL_METHOD,
            "fit_rows": interval.fit.rows,
            "fit_width": interval.fit.width,
            "degrees_of_freedom": interval.degrees_of_freedom,
            "cohort_rows": interval.cohort_rows,
            "level": interval.level,
            "standard_error_basis": "hc3_sandwich_phibar_v_phibar",
            "reported_scalar": FUNCTIONAL_ID,
            "model_identity": wire.export_identity,
        });
        authorize(
            "cohort_interval",
            bytes.to_vec(),
            &body.premises_blake3,
            &body.data_blake3,
            scope,
            vec![ScalarExecution {
                name: FUNCTIONAL_ID.into(),
                point: interval.point,
                interval: (interval.lower, interval.upper),
                basis: calibration_basis(interval.fit.rows, interval.level),
            }],
        )
    }

    /// Consume a measured envelope by replaying its source and comparing the report.
    ///
    /// # Errors
    /// Any source, expected identity, protocol, evidence or scalar mismatch.
    pub fn consume_measured(
        bytes: &[u8],
        expected: &MeasuredExpectation,
        ctx: &ExecutionContext,
    ) -> Result<MeasuredInference, IoError> {
        let wire = decode(bytes, expected)?;
        let measured = Self::measure(&wire.source, ctx)?;
        compare_replay(&wire, &measured)?;
        // The candidate digest binds the envelope to exactly the retained source bytes.
        if expected.candidate_digest
            != crate::identity::digest_canonical(IdentityDomain::Claim, &wire.source).to_hex()
        {
            return Err(refused(
                "measured_inference.artifact_mismatch",
                "retained source digest differs from the envelope source",
            ));
        }
        Ok(measured)
    }
}

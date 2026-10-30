//! Convert [`CausalPosterior`] (internal) ↔ versioned wire artifacts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::VERSION;
use antecedent_estimate::CausalPosterior;
use antecedent_prob::PosteriorQuantityKind;

use crate::container::EncodedArtifact;
use crate::error::IoError;
use crate::posterior::{
    CausalPosteriorWire, PosteriorPreparationWire, PosteriorQuantityWire, encode_posterior_artifact,
};
pub use antecedent_io_core::posterior::decode_causal_posterior_bytes;

/// Which posterior payload to serialize across the FFI / artifact boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PosteriorPayload {
    /// Meta summaries only (mean/sd/quantiles); no draw bytes. Enough for prior hydrate.
    Summary,
    /// Meta + full column-major draws.
    FullDraws,
}

/// Encode a [`CausalPosterior`] to container bytes (Python / tooling).
///
/// Defaults to [`PosteriorPayload::FullDraws`] for Rust tooling parity.
///
/// # Errors
///
/// IO failures.
pub fn encode_causal_posterior_bytes(
    posterior: &CausalPosterior,
    artifact_id: &str,
) -> Result<Vec<u8>, IoError> {
    encode_causal_posterior_bytes_with_payload(posterior, artifact_id, PosteriorPayload::FullDraws)
}

/// Encode a [`CausalPosterior`] with an explicit payload mode.
///
/// # Errors
///
/// IO failures.
pub fn encode_causal_posterior_bytes_with_payload(
    posterior: &CausalPosterior,
    artifact_id: &str,
    payload: PosteriorPayload,
) -> Result<Vec<u8>, IoError> {
    let art = encode_causal_posterior_with_payload(posterior, artifact_id, payload)?;
    let mut buf = Vec::new();
    art.write_to(&mut buf)?;
    Ok(buf)
}

/// Encode a [`CausalPosterior`] with an explicit payload mode.
///
/// # Errors
///
/// IO failures.
pub fn encode_causal_posterior_with_payload(
    posterior: &CausalPosterior,
    artifact_id: &str,
    payload: PosteriorPayload,
) -> Result<EncodedArtifact, IoError> {
    let quantities: Vec<PosteriorQuantityWire> = posterior
        .draws
        .schema
        .quantities
        .iter()
        .map(|q| {
            Ok(match q {
                PosteriorQuantityKind::Coefficient { index, name } => {
                    PosteriorQuantityWire::Coefficient {
                        index: u32::try_from(*index).map_err(|_| IoError::TooLarge)?,
                        name: name.as_ref().map(std::string::ToString::to_string),
                    }
                }
                PosteriorQuantityKind::ResidualVariance => PosteriorQuantityWire::ResidualVariance,
                PosteriorQuantityKind::Effect { name } => {
                    PosteriorQuantityWire::Effect { name: name.to_string() }
                }
                PosteriorQuantityKind::Scalar { name } => {
                    PosteriorQuantityWire::Scalar { name: name.to_string() }
                }
            })
        })
        .collect::<Result<_, IoError>>()?;
    let (draws_encoding, draws): (&str, &[f64]) = match payload {
        PosteriorPayload::FullDraws => ("f64_le_colmajor", posterior.draws.values.as_ref()),
        PosteriorPayload::Summary => ("none", &[]),
    };
    let meta = CausalPosteriorWire {
        quantities,
        n_draws: u32::try_from(posterior.draws.n_draws).map_err(|_| IoError::TooLarge)?,
        mean: posterior.summaries.mean.to_vec(),
        sd: posterior.summaries.sd.to_vec(),
        q025: posterior.summaries.q025.to_vec(),
        q975: posterior.summaries.q975.to_vec(),
        identification: crate::analysis_wire::identification_status_pascal(
            posterior.identification,
        )
        .into(),
        unidentified_mass: posterior.unidentified_mass,
        subsampled_out_mass: posterior.subsampled_out_mass,
        backend_id: posterior.diagnostics.backend_id.to_string(),
        converged: posterior.diagnostics.converged,
        hessian_condition: posterior.diagnostics.hessian_condition,
        draws_encoding: draws_encoding.into(),
        treatment_contrast: posterior.treatment_contrast,
        preparation: posterior.preparation.as_deref().map(|p| PosteriorPreparationWire {
            backend: p.backend.clone(),
            route_bound: p.route_bound.clone(),
            rows_admitted: p.rows_admitted,
            receipt_digest: p.receipt_digest.clone(),
            admitted_bytes: p.admitted_bytes,
            budget_bytes: p.budget_bytes,
            delivery: p.delivery.clone(),
        }),
    };
    encode_posterior_artifact(&meta, draws, artifact_id, VERSION)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posterior::{CausalPosteriorWire, PosteriorQuantityWire};

    #[test]
    fn posterior_export_refuses_narrowing_without_allocating_giant_draws() {
        use antecedent_core::{AssumptionSet, IdentificationStatus};
        use antecedent_prob::{InferenceDiagnostics, PosteriorDraws, PosteriorSchema};
        use std::sync::Arc;
        let schema = PosteriorSchema {
            quantities: Arc::from([PosteriorQuantityKind::Coefficient { index: 0, name: None }]),
        };
        let draws = PosteriorDraws::from_column_major(schema, 2, vec![1.0, 1.0]).unwrap();
        let summaries = draws.summarize();
        let mut posterior = CausalPosterior {
            draws,
            summaries,
            identification: IdentificationStatus::NonparametricallyIdentified,
            prior_sensitivity: None,
            conflict_summary: None,
            diagnostics: InferenceDiagnostics::laplace("test", true, 1, 0.0, 1.0, false, vec![]),
            assumptions: AssumptionSet::new(),
            unidentified_mass: 0.0,
            subsampled_out_mass: 0.0,
            unevaluable_mass: 0.0,
            early_stopped: false,
            treatment_contrast: None,
            preparation: None,
        };
        let bytes = encode_causal_posterior_bytes_with_payload(
            &posterior,
            "normal",
            PosteriorPayload::Summary,
        )
        .unwrap();
        let (normal, _) = decode_causal_posterior_bytes(&bytes).unwrap();
        assert_eq!(normal.n_draws, 2);
        assert!(matches!(
            normal.quantities[0],
            PosteriorQuantityWire::Coefficient { index: 0, .. }
        ));
        if let Ok(large) = usize::try_from(u64::from(u32::MAX) + 1) {
            posterior.draws.n_draws = large + 1;
            assert!(matches!(
                encode_causal_posterior_bytes_with_payload(
                    &posterior,
                    "large-count",
                    PosteriorPayload::Summary
                ),
                Err(IoError::TooLarge)
            ));
            posterior.draws.n_draws = 2;
            posterior.draws.schema.quantities =
                Arc::from([PosteriorQuantityKind::Coefficient { index: large, name: None }]);
            assert!(matches!(
                encode_causal_posterior_bytes_with_payload(
                    &posterior,
                    "large-index",
                    PosteriorPayload::Summary
                ),
                Err(IoError::TooLarge)
            ));
        }
    }

    #[test]
    fn summary_payload_encoding_omits_draw_bytes() {
        // Stored summaries must describe the embedded draws (validate_summaries_against_draws
        // recomputes them with PosteriorDraws::summarize), so use constant per-quantity draws
        // whose mean/sd/quantiles are exact: quantity 0 is 0.1 everywhere, quantity 1 is 2.0.
        let meta = CausalPosteriorWire {
            quantities: vec![
                PosteriorQuantityWire::Coefficient { index: 0, name: Some("intercept".into()) },
                PosteriorQuantityWire::Effect { name: "ate".into() },
            ],
            n_draws: 8192,
            mean: vec![0.1, 2.0],
            sd: vec![0.0, 0.0],
            q025: vec![0.1, 2.0],
            q975: vec![0.1, 2.0],
            identification: "NonparametricallyIdentified".into(),
            unidentified_mass: 0.0,
            subsampled_out_mass: 0.0,
            backend_id: "laplace".into(),
            converged: true,
            hessian_condition: 1.0,
            draws_encoding: "none".into(),
            treatment_contrast: None,
            preparation: None,
        };
        let summary = encode_posterior_artifact(&meta, &[], "summary", VERSION).unwrap();
        let mut summary_bytes = Vec::new();
        summary.write_to(&mut summary_bytes).unwrap();

        let mut full_meta = meta.clone();
        full_meta.draws_encoding = "f64_le_colmajor".into();
        let mut draws = vec![0.1_f64; 8192];
        draws.extend(std::iter::repeat_n(2.0_f64, 8192));
        let full = encode_posterior_artifact(&full_meta, &draws, "full", VERSION).unwrap();
        let mut full_bytes = Vec::new();
        full.write_to(&mut full_bytes).unwrap();
        assert!(summary_bytes.len() < full_bytes.len());
        // Draw payload is omitted under summary encoding (container may compress).
        assert_eq!(
            summary.sections.iter().find(|s| s.id == "posterior.draws").map(|s| s.data.len()),
            Some(0)
        );

        let (wire, decoded_draws) = decode_causal_posterior_bytes(&summary_bytes).unwrap();
        assert!(decoded_draws.is_empty());
        assert_eq!(wire.draws_encoding, "none");
        assert_eq!(wire.n_draws, 8192);
        assert_eq!(wire.mean, vec![0.1, 2.0]);
        assert_eq!(wire.sd, vec![0.0, 0.0]);
    }

    #[test]
    fn posterior_preparation_round_trips_and_is_absent_when_unrecorded() {
        let mut meta = CausalPosteriorWire {
            quantities: vec![PosteriorQuantityWire::Effect { name: "ate".into() }],
            n_draws: 4,
            mean: vec![1.0],
            sd: vec![0.0],
            q025: vec![1.0],
            q975: vec![1.0],
            identification: "NonparametricallyIdentified".into(),
            unidentified_mass: 0.0,
            subsampled_out_mass: 0.0,
            backend_id: "laplace".into(),
            converged: true,
            hessian_condition: 1.0,
            draws_encoding: "f64_le_colmajor".into(),
            treatment_contrast: None,
            preparation: None,
        };
        let draws = [1.0_f64; 4];
        let absent = serde_json::to_string(&meta).unwrap();
        assert!(!absent.contains("preparation"), "an absent receipt must emit no key");
        let artifact = encode_posterior_artifact(&meta, &draws, "without", VERSION).unwrap();
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        let (decoded, _) = decode_causal_posterior_bytes(&bytes).unwrap();
        assert!(decoded.preparation.is_none());

        meta.preparation = Some(PosteriorPreparationWire {
            backend: "in_memory".into(),
            route_bound: "rows_only".into(),
            rows_admitted: Some(4),
            receipt_digest: "ab12".into(),
            admitted_bytes: Some(64),
            budget_bytes: None,
            delivery: Some("finite".into()),
        });
        let artifact = encode_posterior_artifact(&meta, &draws, "with", VERSION).unwrap();
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        let (decoded, _) = decode_causal_posterior_bytes(&bytes).unwrap();
        assert_eq!(decoded.preparation, meta.preparation);
        let legacy: CausalPosteriorWire = serde_json::from_str(&absent).unwrap();
        assert!(legacy.preparation.is_none());
    }
}

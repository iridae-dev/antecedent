//! Source-bound replay for the retained compiler route.

use antecedent_core::{CanonicalWriter, ExecutionContext};
use serde::{Deserialize, Serialize};

use super::{
    FittedLatentSoftIntervention, LatentSoftInterventionDiscovery, LatentSoftInterventionForecast,
    LatentSoftInterventionRequest, cancelled, invalid,
};
use crate::{CausalError, Study};

const MAGIC: &[u8] = b"ANTECEDENT-LATENT-SOFT-INTERVENTION-1";
const MAX_BYTES: usize = 48 * 1024 * 1024;

#[derive(Serialize)]
struct ExportWire<'a> {
    request: &'a LatentSoftInterventionRequest,
    context: &'a [f64],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LoadWire {
    request: LatentSoftInterventionRequest,
    context: Vec<f64>,
}

/// A replayable forecast export and the expectation to retain outside its bytes.
pub struct LatentSoftInterventionExport {
    /// Versioned bytes containing the source observations and forecast context.
    pub bytes: Vec<u8>,
    /// Source, program, forecast context, and output digests, separated by colons.
    pub expectation: String,
}

/// Outcome of a native replay, without a live retained fit.
#[derive(Serialize)]
pub struct ReplayedLatentSoftIntervention {
    /// Discovery re-executed from the same native lowered program.
    pub discovery: LatentSoftInterventionDiscovery,
    /// Forecast re-derived from that fit.
    pub forecast: LatentSoftInterventionForecast,
}

fn context_digest(context: &[f64]) -> String {
    let mut writer = CanonicalWriter::new("antecedent.latent_soft_intervention.replay_context.v1");
    writer.reals(context);
    writer.finish_hex()
}

fn expectation(
    fitted: &FittedLatentSoftIntervention,
    context: &[f64],
    output: &LatentSoftInterventionForecast,
) -> Result<String, CausalError> {
    let mut writer = CanonicalWriter::new("antecedent.latent_soft_intervention.replay.v1");
    writer
        .bytes(fitted.prepared.source.as_bytes())
        .bytes(fitted.prepared.program_identity().as_bytes())
        .bytes(fitted.fit_identity.as_bytes())
        .bytes(&serde_json::to_vec(output).map_err(invalid)?);
    Ok(format!(
        "{}:{}:{}:{}",
        fitted.prepared.source.to_hex(),
        fitted.prepared.program_identity().to_hex(),
        context_digest(context),
        writer.finish_hex(),
    ))
}

impl FittedLatentSoftIntervention {
    /// Export the source and exact forecast context for independent replay.
    ///
    /// # Errors
    /// Unsupported context, a resource refusal, cancellation, or an encoding error.
    pub fn export_forecast(
        &self,
        context: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<LatentSoftInterventionExport, CausalError> {
        let forecast = self.forecast(context, ctx)?;
        let request = &*self.prepared.request;
        // A finite binary64 has a shortest round-trip JSON representation of at most 25 bytes.
        let cells = (request.baseline.len() + request.action.len()) * request.variables.len();
        let bytes_bound = (32 * cells + 65_536).min(MAX_BYTES);
        let _lease = ctx.acquire_admission(0, bytes_bound as u64).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "latent soft intervention export" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
        let mut bytes = MAGIC.to_vec();
        serde_json::to_writer(&mut bytes, &ExportWire { request, context }).map_err(invalid)?;
        cancelled(ctx, "latent soft intervention export")?;
        if bytes.len() > MAX_BYTES {
            return Err(invalid("artifact exceeds the supported byte limit"));
        }
        Ok(LatentSoftInterventionExport {
            bytes,
            expectation: expectation(self, context, &forecast)?,
        })
    }
}

/// Recompile, re-execute and verify the forecast against an independently held identity.
///
/// # Errors
/// Bad encoding, changed source or forecast, invalid declarations, or an execution refusal.
pub fn replay_forecast(
    bytes: &[u8],
    expected: &str,
    ctx: &ExecutionContext,
) -> Result<ReplayedLatentSoftIntervention, CausalError> {
    super::super::lowering::refuse_reentry(ctx)?;
    let mut parts = expected.split(':');
    let [Some(source), Some(program), Some(context), Some(output)] =
        [parts.next(), parts.next(), parts.next(), parts.next()]
    else {
        return Err(invalid("invalid replay expectation"));
    };
    if bytes.len() > MAX_BYTES
        || parts.next().is_some()
        || [source, program, context, output]
            .iter()
            .any(|value| antecedent_core::identity::digest_from_lower_hex(value).is_none())
    {
        return Err(invalid("invalid artifact size or replay expectation"));
    }
    let body = bytes.strip_prefix(MAGIC).ok_or_else(|| invalid("unknown artifact version"))?;
    let _lease =
        ctx.acquire_admission(0, (bytes.len() as u64).saturating_mul(8)).map_err(|error| {
            match error {
                antecedent_core::execution::AdmissionError::Cancelled => {
                    CausalError::Cancelled { stage: "latent soft intervention replay" }
                }
                other => CausalError::Resource { message: other.to_string() },
            }
        })?;
    let wire: LoadWire = serde_json::from_slice(body).map_err(invalid)?;
    if context_digest(&wire.context) != context {
        return Err(invalid("artifact forecast context differs from the expectation"));
    }
    let prepared = Study::prepare_latent_soft_intervention(wire.request, ctx)?;
    if prepared.source.to_hex() != source || prepared.program_identity().to_hex() != program {
        return Err(invalid("artifact source or lowered program differs from the expectation"));
    }
    let fitted = prepared.execute(ctx)?;
    let forecast = fitted.forecast(&wire.context, ctx)?;
    if expectation(&fitted, &wire.context, &forecast)? != expected {
        return Err(invalid(
            "replayed source, program, fit or forecast differs from the expectation",
        ));
    }
    Ok(ReplayedLatentSoftIntervention { discovery: fitted.discovery, forecast })
}

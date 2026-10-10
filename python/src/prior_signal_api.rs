//! Original posterior artifacts through the checked prior/future-signal adapter.
use crate::value_err;
use antecedent::analysis::design_ranking::{
    DesignRankingRequestWire, PriorDeclWire, ProviderDeclWire,
};
use antecedent_design::preposterior::DecisionPrior;
use antecedent_design::prior_signal::{
    CandidateSignalPrior, CandidateSignalSource, NativeSignalFamily, PriorSignalRequest,
    PriorSourceInput, SourceToTarget, TransportPolicyDecl, adapt_prior_to_signal,
};
use antecedent_io::{
    EncodedArtifact, EstimandFingerprint, PosteriorQuantityWire, PriorCatalog, PriorSourceRef,
    TargetDesign, decode_posterior_artifact,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use serde::Deserialize;

const MAX_BYTES: usize = 16 * 1024 * 1024;
const MAX_SOURCES: usize = 64;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceWire {
    artifact_id: String,
    quantity: String,
    source_population: String,
    lineage: Vec<String>,
    observation_ids: Vec<String>,
    weight: f64,
    prior_strength: f64,
    conflict_shrinkage: f64,
    intercept: f64,
    slope: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestWire {
    prior_id: String,
    target_population: String,
    transport_policy_id: Option<String>,
    sources: Vec<SourceWire>,
    resolution: usize,
    candidate_observation_ids: Vec<String>,
    query_kind: String,
    treatment: String,
    outcome: String,
    variables: Vec<String>,
    signal: ProviderDeclWire,
    state: antecedent_io::quantity_wire::ScientificQuantityWire,
}

/// Immutable adapter result: the posterior rows are never supplied back from Python.
#[pyclass(frozen, skip_from_py_object)]
pub(crate) struct CheckedPriorSignal {
    adapted: CandidateSignalPrior,
    family: NativeSignalFamily,
    state: antecedent_io::quantity_wire::ScientificQuantityWire,
    observations: Vec<String>,
    source_digests: Vec<String>,
    #[pyo3(get)]
    diagnostics_json: String,
}

impl CheckedPriorSignal {
    pub(crate) fn bind(&self, request: &mut DesignRankingRequestWire) -> Result<(), String> {
        if !matches!(&request.decision.prior, PriorDeclWire::Draws { states } if states.is_empty())
        {
            return Err(refusal("prior_signal.prior_projection_mismatch", "invalid_argument"));
        }
        for candidate in &request.candidates {
            if candidate.signal.state_quantity != self.state {
                return Err(refusal("prior_signal.state_quantity_mismatch", "invalid_argument"));
            }
            if candidate.signal.prior_id != self.adapted.prior_id {
                return Err(refusal("prior_signal.prior_identity_mismatch", "invalid_argument"));
            }
            let matches = match (&candidate.provider, self.family) {
                (ProviderDeclWire::Binomial, NativeSignalFamily::Binomial) => true,
                (
                    ProviderDeclWire::GaussianMean { noise_variance: actual },
                    NativeSignalFamily::GaussianMean { noise_variance },
                ) => actual.to_bits() == noise_variance.to_bits(),
                _ => false,
            };
            if !matches {
                return Err(refusal(
                    "prior_signal.signal_family_mismatch",
                    "design_signal_invalid",
                ));
            }
        }
        let DecisionPrior::Draws(draws) = &self.adapted.prior else { unreachable!() };
        request.decision.prior = PriorDeclWire::Draws { states: draws.clone() };
        request.prior_observation_ids.extend(self.observations.iter().cloned());
        request.source_digests.extend(self.source_digests.iter().cloned());
        for candidate in &mut request.candidates {
            candidate
                .signal
                .evidence_lineage
                .extend(self.adapted.diagnostics.lineage.iter().cloned());
            candidate.signal.evidence_lineage.sort();
            candidate.signal.evidence_lineage.dedup();
        }
        request.prior_observation_ids.sort();
        request.prior_observation_ids.dedup();
        request.source_digests.sort();
        request.source_digests.dedup();
        Ok(())
    }
}
fn refusal(detail: &str, code: &str) -> String {
    serde_json::json!({"code":code,"stage":"declare","detail":detail,"offending":null,"expected":null,"supplied":null,"remedy":null}).to_string()
}

#[pyfunction]
pub(crate) fn adapt_prior_signal(
    sources: &Bound<'_, PyList>,
    request_json: &str,
) -> PyResult<(Option<CheckedPriorSignal>, Option<String>)> {
    if sources.len() > MAX_SOURCES || request_json.len() > MAX_BYTES {
        return Err(value_err("prior signal declaration exceeds its resource bound"));
    }
    let wire: RequestWire = serde_json::from_str(request_json)
        .map_err(|e| value_err(format!("prior_signal.invalid_request: {e}")))?;
    if wire.sources.len() > MAX_SOURCES {
        return Err(value_err("too many prior signal sources"));
    }
    if wire.state.population_id != wire.target_population {
        return Ok((
            None,
            Some(refusal("prior_signal.target_population_mismatch", "invalid_argument")),
        ));
    }
    let _: antecedent_core::ScientificQuantity =
        wire.state.clone().try_into().map_err(|e| value_err(format!("{e:?}")))?;
    let family = match wire.signal {
        ProviderDeclWire::Binomial => NativeSignalFamily::Binomial,
        ProviderDeclWire::GaussianMean { noise_variance } => {
            NativeSignalFamily::GaussianMean { noise_variance }
        }
        ProviderDeclWire::External(_) => {
            return Ok((
                None,
                Some(refusal("prior_signal.unsupported_signal_provider", "route_not_supported")),
            ));
        }
    };
    let mut refs = Vec::new();
    let mut inputs = Vec::new();
    let mut posterior_diagnostics = Vec::new();
    let mut observations = Vec::new();
    let mut digests = Vec::new();
    let mut total_bytes = 0usize;
    for item in sources.iter() {
        let entry = item.cast::<PyDict>()?;
        let meta = crate::prior_bank::meta_from_dict(
            entry
                .get_item("meta")?
                .ok_or_else(|| value_err("source requires meta"))?
                .cast::<PyDict>()?,
        )?;
        let bytes_object = entry.get_item("artifact")?.filter(|a| !a.is_none());
        let bytes = if let Some(object) = bytes_object {
            let object = object.cast::<PyBytes>()?;
            total_bytes = total_bytes.saturating_add(object.as_bytes().len());
            if total_bytes > MAX_BYTES {
                return Err(value_err("prior artifacts exceed byte bound"));
            }
            Some(object.as_bytes().to_vec())
        } else {
            None
        };
        if let Some(input) = wire.sources.iter().find(|s| s.artifact_id == meta.artifact_id) {
            let Some(bytes) = &bytes else {
                return Ok((
                    None,
                    Some(refusal("prior_signal.posterior_draws_required", "design_signal_invalid")),
                ));
            };
            let artifact = EncodedArtifact::read_from(bytes.as_slice()).map_err(crate::py_err)?;
            if meta.tags.get("population").is_some_and(|p| p != &input.source_population) {
                return Ok((
                    None,
                    Some(refusal("prior_signal.source_population_mismatch", "invalid_argument")),
                ));
            }
            let (posterior, draws) = decode_posterior_artifact(&artifact).map_err(crate::py_err)?;
            if !matches!(
                posterior.identification.as_str(),
                "NonparametricallyIdentified"
                    | "nonparametrically_identified"
                    | "IdentifiedUnderParametricRestrictions"
                    | "identified_under_parametric_restrictions"
            ) {
                return Ok((
                    None,
                    Some(refusal(
                        "prior_signal.original_posterior_unidentified",
                        "design_signal_invalid",
                    )),
                ));
            }
            let indices: Vec<_> = posterior
                .quantities
                .iter()
                .enumerate()
                .filter_map(|(i, q)| match q {
                    PosteriorQuantityWire::Effect { name }
                    | PosteriorQuantityWire::Scalar { name }
                        if name == &input.quantity =>
                    {
                        Some(i)
                    }
                    _ => None,
                })
                .collect();
            if indices.len() != 1 || draws.is_empty() {
                return Ok((
                    None,
                    Some(refusal(
                        "prior_signal.named_state_draws_required",
                        "design_signal_invalid",
                    )),
                ));
            }
            if posterior.unidentified_mass > 0.0
                || posterior.subsampled_out_mass > 0.0
                || !posterior.converged
            {
                return Ok((
                    None,
                    Some(refusal("prior_signal.incomplete_posterior", "design_signal_invalid")),
                ));
            }
            posterior_diagnostics.push(
                serde_json::json!({"artifact_id": input.artifact_id, "posterior": posterior}),
            );
            let n = posterior.n_draws as usize;
            let index = indices[0];
            inputs.push(PriorSourceInput {
                artifact_id: input.artifact_id.clone(),
                source_population: input.source_population.clone(),
                draws: draws[index * n..(index + 1) * n].to_vec(),
                weight: input.weight,
                prior_strength: input.prior_strength,
                conflict_shrinkage: input.conflict_shrinkage,
                lineage: input.lineage.clone(),
                observation_ids: input.observation_ids.clone(),
                source_to_target: SourceToTarget::Affine {
                    intercept: input.intercept,
                    slope: input.slope,
                },
            });
            observations.extend(input.observation_ids.iter().cloned());
            digests.push(blake3::hash(bytes).to_hex().to_string());
        }
        refs.push(match bytes {
            Some(bytes) => PriorSourceRef::with_bytes(meta, bytes),
            None => PriorSourceRef::from_meta(meta),
        });
    }
    // Preserve declared duplicate/missing inputs so the checked adapter issues its own refusal.
    if inputs.len() != wire.sources.len() {
        return Ok((
            None,
            Some(refusal(
                "prior_signal.source_not_in_catalog_or_duplicate",
                "design_signal_invalid",
            )),
        ));
    }
    let target = TargetDesign::new(
        EstimandFingerprint::new(wire.query_kind, wire.treatment, wire.outcome),
        wire.variables,
    );
    let request = PriorSignalRequest {
        prior_id: wire.prior_id,
        sources: inputs,
        signal: CandidateSignalSource::Native(family),
        target_population: wire.target_population,
        transport: wire.transport_policy_id.map(|policy_id| TransportPolicyDecl { policy_id }),
        candidate_observation_ids: wire.candidate_observation_ids,
        resolution: wire.resolution,
    };
    let adapted = match adapt_prior_to_signal(&PriorCatalog::from_sources(refs), &target, &request)
    {
        Ok(adapted) => adapted,
        Err(error) => {
            let r = error.to_refusal();
            return Ok((None, Some(serde_json::json!({"code":r.code,"stage":r.stage,"detail":r.detail,"offending":r.offending,"expected":r.expected,"supplied":r.supplied,"remedy":r.remedy}).to_string())));
        }
    };
    let DecisionPrior::Draws(rows) = &adapted.prior else { unreachable!() };
    if !rows.iter().all(|v| v.is_finite()) || !adapted.diagnostics.pooled_mean.is_finite() {
        return Ok((
            None,
            Some(refusal("prior_signal.nonfinite_mapped_prior", "invalid_argument")),
        ));
    }
    let d = &adapted.diagnostics;
    let diagnostics_json = serde_json::json!({"source_posteriors":posterior_diagnostics,"prior_approximation":"deterministic_resampled_original_posterior_draws","calibration":"unmeasured","source_ids":d.source_ids,"effective_weights":d.effective_weights,"draws_allocated":d.draws_allocated,"pooled_mean":d.pooled_mean,"lineage":d.lineage,"observations_checked":d.observations_checked,"overlapping_observations":d.overlapping_observations,"transport_policy_id":d.transport_policy_id,"source_digest":d.source_digest}).to_string();
    digests.push(d.source_digest.clone());
    // Bind population/maps/declared observation lineage, not only numerical adapter rows.
    digests.push(blake3::hash(request_json.as_bytes()).to_hex().to_string());
    Ok((
        Some(CheckedPriorSignal {
            adapted,
            family,
            state: wire.state,
            observations,
            source_digests: digests,
            diagnostics_json,
        }),
        None,
    ))
}
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CheckedPriorSignal>()?;
    m.add_function(wrap_pyfunction!(adapt_prior_signal, m)?)
}

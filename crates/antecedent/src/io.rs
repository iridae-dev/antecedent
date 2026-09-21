//! Graph interchange and durable artifacts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_io::{
    AnalysisResultConsumption, AnalysisResultContractWire, AssignmentDesignWire, CONTRACT_SECTION,
    ExposureLevelWire, ExposureMappingWire, ExposureProbabilityMethodWire,
    InterferenceEstimateWire, InterferenceFunctionalWire, InterferenceQueryWire,
    NotCertifiedCertificateWire, PopulationFactorWire, RandomizationContrastWire,
    TransportCertificateWire, TransportEffectEstimateWire, TransportFormulaWire,
    TransportIdentificationWire, TransportOverlapDiagnosticWire, TransportQueryWire, admg_from_dot,
    admg_from_gml, admg_from_json, admg_from_networkx_node_link, admg_to_dot, admg_to_gml,
    admg_to_json, admg_to_networkx_node_link, consume_analysis_result, cpdag_from_dot,
    cpdag_from_gml, cpdag_from_json, cpdag_from_networkx_node_link, cpdag_to_dot, cpdag_to_gml,
    cpdag_to_json, cpdag_to_networkx_node_link, dag_from_dot, dag_from_gml, dag_from_json,
    dag_from_networkx_adjacency, dag_from_networkx_node_link, dag_to_dot, dag_to_gml, dag_to_json,
    dag_to_networkx_adjacency, dag_to_networkx_node_link, decode_analysis_result_artifact,
    decode_analysis_result_contract, decode_causal_posterior_bytes,
    encode_analysis_result_artifact, encode_analysis_result_artifact_with_contract,
    encode_causal_posterior, encode_causal_posterior_bytes, interference_estimate_from_wire,
    interference_estimate_to_wire, interference_query_from_wire, interference_query_to_wire,
    pag_from_dot, pag_from_gml, pag_from_json, pag_from_networkx_node_link, pag_to_dot, pag_to_gml,
    pag_to_json, pag_to_networkx_node_link, transport_effect_from_wire, transport_effect_to_wire,
    transport_identification_from_wire, transport_identification_to_wire,
    transport_query_from_wire, transport_query_to_wire,
};

pub use antecedent_io::graph_dot::dag_with_names_from_dot;
pub use antecedent_io::graph_gml::dag_with_names_from_gml;
pub use antecedent_io::graph_json::dag_with_names_from_json;
pub use antecedent_io::graph_mixed::{
    admg_with_names_from_dot, admg_with_names_from_gml, admg_with_names_from_json,
    admg_with_names_from_networkx_node_link, cpdag_with_names_from_dot, cpdag_with_names_from_gml,
    cpdag_with_names_from_json, cpdag_with_names_from_networkx_node_link, pag_with_names_from_dot,
    pag_with_names_from_gml, pag_with_names_from_json, pag_with_names_from_networkx_node_link,
};
pub use antecedent_io::graph_networkx::{
    dag_with_names_from_networkx_adjacency, dag_with_names_from_networkx_node_link,
};

use crate::error::CausalError;

/// Encode a model bundle to durable bytes.
///
/// # Errors
///
/// [`CausalError::Serialization`] on IO failures.
pub fn encode_model_bundle_bytes(
    input: &antecedent_io::ModelBundleEncode<'_>,
) -> Result<Vec<u8>, CausalError> {
    let art = antecedent_io::encode_model_bundle(input).map_err(CausalError::from)?;
    let mut buf = Vec::new();
    art.write_to(&mut buf).map_err(CausalError::from)?;
    Ok(buf)
}

/// Decode a model bundle from durable bytes (migrates format if needed).
///
/// # Errors
///
/// [`CausalError::Serialization`] on IO failures.
pub fn decode_model_bundle_bytes(bytes: &[u8]) -> Result<antecedent_io::ModelBundle, CausalError> {
    let art = antecedent_io::read_and_migrate(bytes).map_err(CausalError::from)?;
    antecedent_io::decode_model_bundle(&art).map_err(CausalError::from)
}

/// Hydrate a coefficient [`antecedent_prob::PriorSet`] from posterior artifact bytes.
///
/// Uses per-coefficient posterior means and SDs (identical-subspace mapping).
/// Effect columns are ignored. Prefer [`hydrate_prior_from_posterior_bytes`](crate::inference::hydrate_prior_from_posterior_bytes) when
/// a heterogeneous mapping is required.
///
/// # Errors
///
/// Decode failures or hydrate failures (no coefficients / non-finite summaries).
pub fn prior_set_from_posterior_bytes(
    bytes: &[u8],
) -> Result<antecedent_prob::PriorSet, CausalError> {
    use std::sync::Arc;

    use antecedent_estimate::hydrate_prior_from_quantity_summaries;
    use antecedent_io::PosteriorQuantityWire;
    use antecedent_prob::PosteriorQuantityKind;

    let (wire, _) = decode_causal_posterior_bytes(bytes)?;
    let quantities: Vec<PosteriorQuantityKind> = wire
        .quantities
        .iter()
        .map(|q| match q {
            PosteriorQuantityWire::Coefficient { index, name } => {
                PosteriorQuantityKind::Coefficient {
                    index: *index as usize,
                    name: name.as_ref().map(|s| Arc::<str>::from(s.as_str())),
                }
            }
            PosteriorQuantityWire::ResidualVariance => PosteriorQuantityKind::ResidualVariance,
            PosteriorQuantityWire::Effect { name } => {
                PosteriorQuantityKind::Effect { name: Arc::from(name.as_str()) }
            }
            PosteriorQuantityWire::Scalar { name } => {
                PosteriorQuantityKind::Scalar { name: Arc::from(name.as_str()) }
            }
        })
        .collect();
    hydrate_prior_from_quantity_summaries(&quantities, &wire.mean, &wire.sd, None)
        .map_err(CausalError::from)
}

/// Portable checked transport proof and supplied exact-law records.
pub use antecedent_io::{exact_law_wire::ExactLawWire, transport_proof::TransportProofWire};
